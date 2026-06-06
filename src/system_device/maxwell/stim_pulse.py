"""
Maxwell 双相脉冲序列构造 helper。

参考 Maxwell 官方 examples/python/stimulate.html 的 create_stim_pulse 模板：
  - 双相 DAC：电压模式下放大器反相，DAC 序列须先减后加才能在电极端产生
    正相先脉冲（与 MCS [+a, -a, 0] 直接电压描述等价）
  - 1 sample = 50 μs，恒定，与 sampling rate 无关
  - DAC 中点 512（10-bit DAC，0-1023），1 LSB 通过 mx.query_DAC_lsb_mV()
    动态查询，禁止硬编码
  - mx.Event 在 sequence 内插入 frame 元数据标记，与紧跟的 DAC 同帧
  - mx.clear_events 在 seq.send 之前清一次（启动序列残留的事件不污染）

文档锚点：
  - section_api/subsections/api_python.html#maxlab.sequence.Sequence
  - section_api/subsections/api_python.html#maxlab.chip.DAC
  - section_api/subsections/api_python.html#maxlab.system.DelaySamples
  - section_api/subsections/api_python.html#maxlab.system.Event
  - examples/python/stimulate.html (create_stim_pulse line 633-681)
"""

DAC_CENTER = 512
SAMPLE_DURATION_US = 50  # 1 sample = 50 μs，硬件常数
DEFAULT_INITIAL_DELAY = 100  # samples = 5 ms，官方推荐值，防 first command 早于 download 完成


# mx.Event user_id 命名空间。要求 1 ≤ id ≤ 2^20 - 1, ≠ 0。
# 与 MCS syncoutdata 的 5/6/7/8 << 2 / (1/2/3/4 << 2) - 1 等编码语义对齐。
EVENT_USER_ID_ENV_LEFT = 1
EVENT_USER_ID_ENV_RIGHT = 2
EVENT_USER_ID_PUNISH_LEFT = 3
EVENT_USER_ID_PUNISH_RIGHT = 4
EVENT_USER_ID_REWARD_LEFT = 5
EVENT_USER_ID_REWARD_RIGHT = 6
EVENT_USER_ID_REWARD_BOTH = 7
EVENT_USER_ID_MPC_LEFT = 8
EVENT_USER_ID_MPC_RIGHT = 9


def query_dac_lsb_mv():
    """读取当前固件的 DAC LSB（mV/bit）。每个会话启动后调一次缓存。"""
    import maxlab as mx
    return float(mx.query_DAC_lsb_mV())


def uv_to_dac_bits(uv, dac_lsb_mv):
    """μV 振幅 → DAC bits 偏移。后续与 512 中点相加 / 相减。"""
    return int(round((uv / 1000.0) / dac_lsb_mv))


def us_to_samples(us):
    """μs → DelaySamples 数。50 μs/sample，向下不能为 0。"""
    if us <= 0:
        return 0
    return max(1, int(round(us / SAMPLE_DURATION_US)))


def build_biphasic_pulse_train(
    amp_array_uv,
    dur_array_us,
    dac_lsb_mv,
    well_id=0,
    user_id=1,
    label_prefix="",
    initial_delay=DEFAULT_INITIAL_DELAY,
):
    """
    把 MCS 风格 (amp[], dur[]) 数组构造成 mx.Sequence。

    MCS 端 update_record_stimulation 对每个脉冲生成三段：
        amp = [+a, -a, 0]            (电极端电压)
        dur = [phase, phase, gap]    (μs)
    多个脉冲拼接得到 amp_array / dur_array。set_sti_signal 用四段 slot 模式：
        amp = [a1, 0, a2, 0]
        dur = [d1, d2, d3, gap_us]

    本函数遍历 (amp, dur) 配对，逐段输出：
        a > 0:  DAC(0, 512 - amp_dac)   电极端 +a
        a < 0:  DAC(0, 512 + amp_dac)   电极端 -a
        a == 0: DAC(0, 512)             电极端 0
    每段后跟 DelaySamples(d/50)。

    在每个脉冲的真实起点（首个非零段，不论正/负相先）前插一条 mx.Event 标记，
    使「负相先」波形（reward / set_sti_signal 配置）的标记也对齐脉冲真实起点。
    序列收尾追加 DAC(0, 512) 防 DAC 残留非零值。

    Parameters
    ----------
    amp_array_uv : sequence of int
        每段振幅（μV）。正/负/零三态。
    dur_array_us : sequence of int
        每段持续时间（μs）。长度与 amp_array_uv 一致。
    dac_lsb_mv : float
        当前 DAC LSB（mV/bit），由 query_dac_lsb_mv() 提供。
    well_id : int
        Maxwell well 编号。MaxOne 固定 0。
    user_id : int
        mx.Event user_id（角色标识，见模块顶部常量）。
    label_prefix : str
        Event properties 的语义前缀。
    initial_delay : int
        Sequence(initial_delay=...)，单位 samples。100 = 5 ms（官方默认）。

    Returns
    -------
    maxlab.sequence.Sequence
        已 append 完所有命令、未 send 的 Sequence 对象。
    """
    import maxlab as mx

    if len(amp_array_uv) != len(dur_array_us):
        raise ValueError(
            "amp_array_uv ({}) and dur_array_us ({}) length mismatch.".format(
                len(amp_array_uv), len(dur_array_us)
            )
        )

    seq = mx.Sequence(initial_delay=initial_delay)
    pulse_idx = 0
    in_pulse = False   # 是否处于一个双相脉冲内部

    for i, (a_uv_raw, d_us_raw) in enumerate(zip(amp_array_uv, dur_array_us)):
        a_uv = int(a_uv_raw)
        d_us = int(d_us_raw)

        # 在每个脉冲的【真实起点】（首个非零段，不论正相/负相先）插一条 mx.Event。
        # 旧逻辑只在 a_uv>0 处插：对「负相先」波形（reward 的 [-a,+a,0]、set_sti_signal
        # 配置的 [a1<0,0,a2>0,gap]）会把标记打到第二相（晚 ~一个相位），不对齐真实起点。
        if a_uv != 0 and not in_pulse:
            pulse_idx += 1
            # mx.Event 的 properties 必须是「空格分隔的 key value 成对」(偶数个 token)。
            # 原 label 对单词 prefix（punish_left / reward_left）会得到奇数 token
            # （"punish_left pulse_1 amp_500000uV" = 3 个），服务器按 key-value 解析时整条
            # Event 被丢弃 → punish/reward/MPC 标签全丢；env 的 prefix "env_left freq_22"
            # 恰好凑成偶数 token 才得以保留。这里统一成 6-token(3 对) key-value，prefix 内空格
            # 转 _，保证所有刺激类型都能写进记录。eventid(=user_id) 仍是权威类型标识。
            label = "tag {} pulse {} amp_uV {}".format(
                (str(label_prefix).replace(" ", "_") or "stim"), pulse_idx, a_uv
            )
            seq.append(mx.Event(well_id, 1, user_id, label))
            in_pulse = True

        if a_uv > 0:
            amp_dac = uv_to_dac_bits(a_uv, dac_lsb_mv)
            seq.append(mx.DAC(0, DAC_CENTER - amp_dac))
        elif a_uv < 0:
            amp_dac = uv_to_dac_bits(-a_uv, dac_lsb_mv)
            seq.append(mx.DAC(0, DAC_CENTER + amp_dac))
        else:
            seq.append(mx.DAC(0, DAC_CENTER))

        # 只有「带时长的零段」（真实相间/脉冲间隙）才结束当前脉冲；时长 0 的占位零段
        # （如 set_sti_signal duration_2=0 的中间段）不结束，避免把一个双相脉冲拆成两个 Event。
        if a_uv == 0 and d_us > 0:
            in_pulse = False

        n_samples = us_to_samples(d_us)
        if n_samples > 0:
            seq.append(mx.DelaySamples(n_samples))

    # 收尾回零，防序列结束时 DAC 仍偏离中点
    seq.append(mx.DAC(0, DAC_CENTER))

    return seq


def build_pulse_train_from_freq(
    freq_hz,
    amplitude_uv,
    phase_us,
    dac_lsb_mv,
    well_id=0,
    user_id=1,
    label_prefix="",
    initial_delay=DEFAULT_INITIAL_DELAY,
    polarity="positive_first",
):
    """
    频率 → 脉冲序列。完整复现 MCS update_record_stimulation 与 set_reward_sti
    的脉冲数与时序：

        n_pulses = int(freq_hz * 0.1 + 0.5)   # 100 ms 内的脉冲数
        period_us = 1_000_000 / freq_hz
        每脉冲三段：[a, -a, 0] / [phase, phase, period - 2*phase]   (positive_first)
                  或 [-a, a, 0] / [phase, phase, period - 2*phase]   (negative_first，奖励刺激)

    Parameters
    ----------
    freq_hz : int or float
        刺激频率（Hz）。
    amplitude_uv : int
        每相幅度的绝对值（μV）。
    phase_us : int
        每相持续（μs）。MCS 端固定 200。
    dac_lsb_mv : float
        当前 DAC LSB。
    polarity : {"positive_first", "negative_first"}
        电极端波形相序。环境 / 惩罚走 positive_first（与 MCS env / punish 一致），
        奖励走 negative_first（与 MCS set_reward_sti 一致）。
    其余参数同 build_biphasic_pulse_train。
    """
    if freq_hz <= 0:
        raise ValueError("freq_hz must be > 0, got {}".format(freq_hz))

    n_pulses = int(freq_hz * 0.1 + 0.5)
    if n_pulses < 1:
        n_pulses = 1

    period_us = 1_000_000.0 / freq_hz
    gap_us = int(round(period_us - 2 * phase_us))
    if gap_us < 0:
        raise ValueError(
            "gap_us {} < 0 for freq={}, phase={}us. Reduce phase or freq.".format(
                gap_us, freq_hz, phase_us
            )
        )

    amp = []
    dur = []
    if polarity == "positive_first":
        for _ in range(n_pulses):
            amp.extend([amplitude_uv, -amplitude_uv, 0])
            dur.extend([phase_us, phase_us, gap_us])
    elif polarity == "negative_first":
        for _ in range(n_pulses):
            amp.extend([-amplitude_uv, amplitude_uv, 0])
            dur.extend([phase_us, phase_us, gap_us])
    else:
        raise ValueError("polarity must be 'positive_first' or 'negative_first'")

    return build_biphasic_pulse_train(
        amp_array_uv=amp,
        dur_array_us=dur,
        dac_lsb_mv=dac_lsb_mv,
        well_id=well_id,
        user_id=user_id,
        label_prefix=label_prefix,
        initial_delay=initial_delay,
    )


def estimate_sequence_duration_seconds(amp_array_uv, dur_array_us,
                                       initial_delay=DEFAULT_INITIAL_DELAY):
    """估算 sequence 完整发放耗时（秒），用于 send 后的 sleep 控制。"""
    total_samples = initial_delay
    for d_us in dur_array_us:
        total_samples += us_to_samples(int(d_us))
    return total_samples * SAMPLE_DURATION_US / 1_000_000.0
