"""
Maxwell Stimulation 角色（Phase B2 真实刺激发放）。

业务逻辑层完全复现 MCS Stimulation：
  - update_record_stimulation(left, right) 频率→脉冲转换 + 左右决策
  - 100 Hz / 75 mV / 100 ms 双相奖励刺激参数
  - update_stimulation_left/right 用 set_sti_signal 解析的惩罚波形
  - update_record_stimulation_dynamic_model_left/right 接收 MPC 数组
  - 5 Hz slot + duty cycle + 最小 burst 长度 + 随机 burst 起点 mask 生成

SDK 替换层：MCS STG 调用（PrepareAndSendData / SendStart / channel_map /
ElectrodeMode / DacMux / Enable / BlankingEnable / AmplifierProtection）全部
替换为 Maxwell 的 mx.Sequence + mx.DAC + mx.DelaySamples + mx.Event +
StimulationUnit.connect 切换。

电极角色约定（GUI 入口已固化）：
  stim_electrodes[0] = 左轮，stim_electrodes[1] = 右轮。
  左右切换通过 stim_pool.activate / deactivate（即 StimulationUnit.connect(bool)）
  完成；同一时刻只允许一侧 unit connect=True，避免共享 DAC0 串扰。
  左右奖励同时发放时两 unit 同时 connect=True，共享 DAC0 序列。

文档锚点（每次涉及 API 调用必查）：
  - section_api/subsections/api_python.html#maxlab.sequence.Sequence
  - section_api/subsections/api_python.html#maxlab.chip.DAC
  - section_api/subsections/api_python.html#maxlab.chip.StimulationUnit (connect / dac_source)
  - section_api/subsections/api_python.html#maxlab.system.Event
  - section_api/subsections/api_python.html#maxlab.system.DelaySamples
  - examples/python/stimulate.html (create_stim_pulse, sequential 模式样例)
"""

import random

from .stim_pool import StimPool
from . import stim_pulse


# 奖励刺激默认参数（与 MCS set_reward_sti 一致：100 Hz / 75 mV / 100 ms）
REWARD_FREQ_HZ = 100
REWARD_AMPLITUDE_UV = 75_000   # 75 mV → μV
REWARD_PHASE_US = 200          # 双相每相 200 μs

# 环境编码刺激默认参数（与 MCS update_record_stimulation 一致：amp = 75 mV / phase = 200 μs）
ENV_AMPLITUDE_UV = 75_000
ENV_PHASE_US = 200

# 默认惩罚刺激参数（仅当 set_sti_signal 未被调用时的兜底，避免发放空 sequence）
DEFAULT_PUNISH_FREQ_HZ = 5
DEFAULT_PUNISH_AMPLITUDE_UV = 75_000
DEFAULT_PUNISH_PHASE_US = 200


class StimulationMaxwell(object):
    """
    Maxwell 刺激角色。两电极（左/右轮）固定方案。

    生命周期：
    1. __init__()                       构造期不触碰 maxlab
    2. set_recording(rec)               与 RecordingMaxwell 关联
    3. attach_stim_pool(stim_pool)      由 MaxwellSystem.start_session 注入
    4. set_role_units(left, right)      绑定左右轮电极 / unit_id
       或 set_role_mapping({"left_wheel": .., "right_wheel": ..})
    5. initial_device()                 标记就绪 + 默认全部 unit 关闭
    6. set_sti_signal(sig)              解析惩罚 / 环境波形（缓存 amp/dur 数组）
    7. update_*                         运行时调用，发放刺激
    8. disconnect()                     由 MaxwellSystem.stop_connect 触发
    """

    def __init__(self):
        self.recording = None
        self.sti_sig = None
        self.stim_pool = None

        # 电极 / unit 角色映射（attach_stim_pool 后注入）
        self.left_electrode = None
        self.right_electrode = None
        self.left_unit_id = None
        self.right_unit_id = None
        self.role_to_electrode = {}

        self._connected = False
        self._dac_lsb_mv = None       # query_DAC_lsb_mV 缓存

        # set_sti_signal 解析得到的惩罚波形（μV / μs）
        self.amplitude_uv = None      # list[int]
        self.duration_us = None       # list[int]

        # set_reward_sti 计算得到的奖励波形（μV / μs）— __init__ 即装好默认值
        self.amp_reward_uv = None
        self.dur_reward_us = None
        self.set_reward_sti()

    # ---------- 角色注入 / 设备就绪 ----------

    def set_recording(self, recording):
        """与 MCS Stimulation 同名方法。建立刺激与记录对象的关联。"""
        self.recording = recording

    def attach_stim_pool(self, stim_pool):
        """
        由 MaxwellSystem 在 connect() 完成后注入已 route + power_up 的 StimPool。

        attach 时一次性：
          1. 缓存 DAC LSB（避免后续每次 update_* 跨进程查询）
          2. 清空 server 端 event buffer 一次（覆盖 route_and_power_up 阶段
             残留事件；之后 update_* 不再 clear，让各次发放的 mx.Event 标记
             保留下来供事后从 frame 元数据回看）
        """
        if not isinstance(stim_pool, StimPool):
            raise TypeError("attach_stim_pool requires a StimPool instance.")
        self.stim_pool = stim_pool
        self._dac_lsb_mv = stim_pulse.query_dac_lsb_mv()
        print("[STIM] attach_stim_pool: DAC LSB cached = {:.4f} mV/bit".format(self._dac_lsb_mv))

        import maxlab as mx
        mx.clear_events()
        print("[STIM] attach_stim_pool: mx.clear_events() — buffer cleared once after route_and_power_up")

    def set_role_units(self, left_electrode, right_electrode):
        """
        绑定左右轮电极。两个电极都必须已在 stim_pool 候选池中。
        attach_stim_pool 必须先调。
        """
        if self.stim_pool is None:
            raise RuntimeError(
                "set_role_units requires stim_pool attached first."
            )
        self.left_electrode = int(left_electrode)
        self.right_electrode = int(right_electrode)
        if self.left_electrode == self.right_electrode:
            raise ValueError(
                "left_electrode and right_electrode must differ; got {} for both".format(
                    self.left_electrode
                )
            )
        self.left_unit_id = self.stim_pool.get_unit(self.left_electrode)
        self.right_unit_id = self.stim_pool.get_unit(self.right_electrode)
        self.role_to_electrode = {
            "left_wheel": self.left_electrode,
            "right_wheel": self.right_electrode,
        }
        print(
            "Maxwell stim roles: left_wheel electrode={} unit={}, "
            "right_wheel electrode={} unit={}".format(
                self.left_electrode, self.left_unit_id,
                self.right_electrode, self.right_unit_id,
            )
        )

    def set_role_mapping(self, role_to_electrode):
        """
        外部 role_mapping 注入入口。如果 mapping 含 left_wheel / right_wheel，
        等价于 set_role_units；否则只缓存 mapping，不绑定 unit。
        """
        if not role_to_electrode:
            return
        left = role_to_electrode.get("left_wheel")
        right = role_to_electrode.get("right_wheel")
        if left is not None and right is not None:
            self.set_role_units(left, right)
        else:
            self.role_to_electrode = dict(role_to_electrode)

    def initial_device(self):
        """
        标记设备就绪 + 默认把两个 unit 都断开（connect=False），让左右刺激不串扰。
        如果未绑定左右轮 unit，退化为 recording-only 模式（update_* 直接 return）。
        """
        if self.left_unit_id is None or self.right_unit_id is None:
            print("Maxwell stim: no left/right unit bound; running recording-only mode.")
            self._connected = False
            return
        self._set_active_only(active_unit_ids=set())
        self._connected = True
        print("Maxwell stimulation device initialized (left/right wheel ready).")

    def set_sti_signal(self, sig):
        """
        解析刺激配置对象。完整复用 MCS set_sti_signal 业务逻辑：
          5 Hz slot + duty cycle 0.6 + min burst length 2 + 随机 burst 起点 → mask
          每 slot 输出 [a1, 0, a2, 0] / [d1, d2, d3, gap]
            （gap = slot_us - pulse_us，保证每 slot 总长度 200 ms）
          mask = 0 时输出 [0, 0, 0, 0] / [d1, d2, d3, gap] 占据空 slot

        sig 字段（与 MCS Stimulation 一致）：
          sig.para         list[dict] 单元参数：amplitude_1/_2 (mV)，
                            duration_1/_2/_3 (μs)，cycles (slot 数)，ISI (μs，未用)
          sig.sti_para     dict，含 reapeat_times

        sig.stimulating_list[0/1] 在 Maxwell 路径下被忽略：左右轮电极
        由 set_role_units 提前注入，不通过 sig 重新配置。
        """
        self.sti_sig = sig
        parameter = sig.para
        sti_para = sig.sti_para

        duty_cycle = 0.6
        min_burst_len = 2
        freq_hz = 5
        random_mode = True

        rng = random.Random(None)

        amp = []
        dur = []
        slot_us = int(round(1_000_000 / freq_hz))   # 200_000 μs / slot

        for _ in range(sti_para["reapeat_times"]):
            for tp in parameter:
                n_slots = int(tp["cycles"])

                if random_mode:
                    mask = _make_burst_mask(
                        n_slots=n_slots,
                        duty=duty_cycle,
                        min_len=min_burst_len,
                        rng=rng,
                    )
                else:
                    mask = [1] * n_slots

                d1 = int(tp["duration_1"])
                d2 = int(tp["duration_2"])
                d3 = int(tp["duration_3"])
                pulse_us = d1 + d2 + d3
                if pulse_us >= slot_us:
                    raise ValueError(
                        "Pulse duration ({} μs) must be < slot_us ({} μs).".format(
                            pulse_us, slot_us
                        )
                    )
                gap_us = slot_us - pulse_us

                a1_uv = int(tp["amplitude_1"] * 1000)
                a2_uv = int(tp["amplitude_2"] * 1000)

                for j in range(n_slots):
                    if mask[j] == 1:
                        amp.extend([a1_uv, 0, a2_uv, 0])
                    else:
                        amp.extend([0, 0, 0, 0])
                    dur.extend([d1, d2, d3, gap_us])

        self.amplitude_uv = amp
        self.duration_us = dur
        self.set_reward_sti()
        print("Maxwell stim signal parsed: {} slot segments cached.".format(
            len(amp) // 4
        ))

    def set_reward_sti(self):
        """
        奖励刺激：100 Hz / 75 mV / 100 ms 双相，参数与 MCS set_reward_sti 完全一致。
        电极端波形：负相先（[-a, +a, 0]），与环境 / 惩罚的正相先（[+a, -a, 0]）
        相对，便于事后从 frame 元数据区分两类刺激。
        """
        n_pulses = int(REWARD_FREQ_HZ * 0.1)            # 10 个脉冲
        period_us = int(1_000_000 / REWARD_FREQ_HZ)     # 10_000 μs
        gap_us = period_us - 2 * REWARD_PHASE_US        # 9_600 μs

        amp = []
        dur = []
        for _ in range(n_pulses):
            amp.extend([-REWARD_AMPLITUDE_UV, REWARD_AMPLITUDE_UV, 0])
            dur.extend([REWARD_PHASE_US, REWARD_PHASE_US, gap_us])

        self.amp_reward_uv = amp
        self.dur_reward_us = dur

    def disconnect(self):
        """容错关闭。stim_pool 由 MaxwellSystem 统一 cleanup，本类不重复处理。"""
        self._connected = False

    # ---------- 内部工具 ----------

    def _set_active_only(self, active_unit_ids):
        """
        把 stim_pool 的激活集合切换为指定 unit_id 集合。
        通过 stim_pool.activate / deactivate（StimulationUnit.connect 通道）实现。

        差集优化：currently_active 与 target 已相同时，跳过所有 connect 切换；
        每次 connect/disconnect 都会引发 ADC 短暂伪迹（用户实测确认），所以
        日志逐项打印决策过程，便于从波形伪迹反向定位。
        """
        if self.stim_pool is None:
            raise RuntimeError("stim_pool not attached")

        target_units = set(int(u) for u in active_unit_ids)

        # unit_id → electrode_id 反查表（stim_pool 内部按 electrode 索引）
        unit_to_electrode = {
            uid: ele for ele, uid in self.stim_pool._electrode_to_unit.items()
        }

        target_electrodes = set()
        for uid in target_units:
            if uid not in unit_to_electrode:
                raise RuntimeError(
                    "unit {} not in stim_pool (have units {})".format(
                        uid, list(unit_to_electrode.keys())
                    )
                )
            target_electrodes.add(unit_to_electrode[uid])

        currently_active = self.stim_pool.get_active_electrodes()
        to_deactivate = currently_active - target_electrodes
        to_activate = target_electrodes - currently_active

        print("[STIM] _set_active_only: currently_active={} target={} "
              "to_deactivate={} to_activate={}".format(
                  sorted(currently_active), sorted(target_electrodes),
                  sorted(to_deactivate), sorted(to_activate)
              ))

        if not to_deactivate and not to_activate:
            print("[STIM] _set_active_only: no change — skipping all connect toggles")
            return

        if to_deactivate:
            self.stim_pool.deactivate(list(to_deactivate))
        if to_activate:
            self.stim_pool.activate(list(to_activate))

    def _send_sequence(self, seq):
        """
        发放 sequence。Sequence.send() 返回 Sequence 自身（不是 'OK' 字符串），
        因此不做 check_send_ok 校验。

        **不在此处 mx.clear_events**：clear_events 已在 attach_stim_pool
        阶段做过一次（覆盖启动 + route_and_power_up 残留），之后保留各次
        update_* 写入的 mx.Event 标记，让事后从 frame 元数据可完整回看刺激历史。
        """
        print("[STIM] _send_sequence: seq.send() (sequence dispatch to MaxHub)")
        seq.send()

    def _send_pulse_train_freq(self, freq_hz, amplitude_uv, phase_us,
                                target_unit_ids, user_id, label_prefix,
                                polarity="positive_first"):
        """
        构造频率→脉冲序列并发到指定 unit 集合。
        发送前 _set_active_only 切换激活集合，确保不串扰。
        """
        seq = stim_pulse.build_pulse_train_from_freq(
            freq_hz=freq_hz,
            amplitude_uv=amplitude_uv,
            phase_us=phase_us,
            dac_lsb_mv=self._dac_lsb_mv,
            user_id=user_id,
            label_prefix=label_prefix,
            polarity=polarity,
        )
        self._set_active_only(set(target_unit_ids))
        self._send_sequence(seq)

    def _send_segment_array(self, amp_array_uv, dur_array_us,
                             target_unit_ids, user_id, label_prefix):
        """
        构造 (amp[], dur[]) 数组的 Sequence 并发到指定 unit 集合。
        """
        if not amp_array_uv:
            print("Maxwell stim: amp/dur cache empty, skipping ({}).".format(label_prefix))
            return

        seq = stim_pulse.build_biphasic_pulse_train(
            amp_array_uv=amp_array_uv,
            dur_array_us=dur_array_us,
            dac_lsb_mv=self._dac_lsb_mv,
            user_id=user_id,
            label_prefix=label_prefix,
        )
        self._set_active_only(set(target_unit_ids))
        self._send_sequence(seq)

    # ---------- 环境编码主路径 ----------

    def update_record_stimulation(self, left, right):
        """
        左右传感器频率（Hz）经编码后传入。决策与 MCS 一致：
          left ≥ right → 左轮发放（频率 = left）
          else        → 右轮发放（频率 = right）
        每侧脉冲数 = int(freq * 0.1 + 0.5)，振幅 75 mV，每相 200 μs，电极端正相先。
        """
        if not self._connected:
            return
        decision = "left" if left >= right else "right"
        print("[STIM] update_record_stimulation(left={}, right={}) -> {}".format(
            left, right, decision
        ))

        if left >= right:
            self._send_pulse_train_freq(
                freq_hz=left,
                amplitude_uv=ENV_AMPLITUDE_UV,
                phase_us=ENV_PHASE_US,
                target_unit_ids={self.left_unit_id},
                user_id=stim_pulse.EVENT_USER_ID_ENV_LEFT,
                label_prefix="env_left freq_{}".format(int(left)),
                polarity="positive_first",
            )
        else:
            self._send_pulse_train_freq(
                freq_hz=right,
                amplitude_uv=ENV_AMPLITUDE_UV,
                phase_us=ENV_PHASE_US,
                target_unit_ids={self.right_unit_id},
                user_id=stim_pulse.EVENT_USER_ID_ENV_RIGHT,
                label_prefix="env_right freq_{}".format(int(right)),
                polarity="positive_first",
            )

    # ---------- 惩罚刺激 ----------

    def update_stimulation_left(self):
        """左侧惩罚刺激（碰撞触发）。波形来自 set_sti_signal 缓存。"""
        if not self._connected:
            return
        print("[STIM] update_stimulation_left() — punish left")
        if not self.amplitude_uv:
            self._send_default_punish(
                target_unit_id=self.left_unit_id,
                user_id=stim_pulse.EVENT_USER_ID_PUNISH_LEFT,
                label_prefix="punish_left default",
            )
            return
        self._send_segment_array(
            amp_array_uv=self.amplitude_uv,
            dur_array_us=self.duration_us,
            target_unit_ids={self.left_unit_id},
            user_id=stim_pulse.EVENT_USER_ID_PUNISH_LEFT,
            label_prefix="punish_left",
        )

    def update_stimulation_right(self):
        """右侧惩罚刺激（碰撞触发）。"""
        if not self._connected:
            return
        print("[STIM] update_stimulation_right() — punish right")
        if not self.amplitude_uv:
            self._send_default_punish(
                target_unit_id=self.right_unit_id,
                user_id=stim_pulse.EVENT_USER_ID_PUNISH_RIGHT,
                label_prefix="punish_right default",
            )
            return
        self._send_segment_array(
            amp_array_uv=self.amplitude_uv,
            dur_array_us=self.duration_us,
            target_unit_ids={self.right_unit_id},
            user_id=stim_pulse.EVENT_USER_ID_PUNISH_RIGHT,
            label_prefix="punish_right",
        )

    # ---------- 奖励刺激 ----------

    def update_stimulation_left_reward(self):
        """左侧奖励刺激（与 MCS update_stimulation_stg1_reward 等价）。"""
        if not self._connected:
            return
        print("[STIM] update_stimulation_left_reward() — reward left")
        self._send_segment_array(
            amp_array_uv=self.amp_reward_uv,
            dur_array_us=self.dur_reward_us,
            target_unit_ids={self.left_unit_id},
            user_id=stim_pulse.EVENT_USER_ID_REWARD_LEFT,
            label_prefix="reward_left",
        )

    def update_stimulation_right_reward(self):
        """右侧奖励刺激。"""
        if not self._connected:
            return
        print("[STIM] update_stimulation_right_reward() — reward right")
        self._send_segment_array(
            amp_array_uv=self.amp_reward_uv,
            dur_array_us=self.dur_reward_us,
            target_unit_ids={self.right_unit_id},
            user_id=stim_pulse.EVENT_USER_ID_REWARD_RIGHT,
            label_prefix="reward_right",
        )

    def update_stimulation_left_right_reward(self):
        """
        双侧同时奖励刺激（与 MCS update_stimulation_stg1_stg2_reward 等价）。
        两 unit 同时 connect=True，共享 DAC0 → 同一 sequence 同步驱动两侧。
        """
        if not self._connected:
            return
        print("[STIM] update_stimulation_left_right_reward() — reward both")
        self._send_segment_array(
            amp_array_uv=self.amp_reward_uv,
            dur_array_us=self.dur_reward_us,
            target_unit_ids={self.left_unit_id, self.right_unit_id},
            user_id=stim_pulse.EVENT_USER_ID_REWARD_BOTH,
            label_prefix="reward_both",
        )

    # ---------- MPC 动力学模型路径 ----------

    def update_record_stimulation_dynamic_model_left(self, ampli, duri):
        """
        MPC 左路径。MCS 端约定 ampli/duri 是嵌套数组 [[a1, a2, ...]]，
        ampli[0] 是 amp 序列（μV），duri[0] 是 dur 序列（μs）。
        """
        if not self._connected:
            return
        print("[STIM] update_record_stimulation_dynamic_model_left() — MPC left")
        amp = list(ampli[0])
        dur = list(duri[0])
        self._send_segment_array(
            amp_array_uv=amp,
            dur_array_us=dur,
            target_unit_ids={self.left_unit_id},
            user_id=stim_pulse.EVENT_USER_ID_MPC_LEFT,
            label_prefix="mpc_left",
        )

    def update_record_stimulation_dynamic_model_right(self, ampli, duri):
        """MPC 右路径。"""
        if not self._connected:
            return
        print("[STIM] update_record_stimulation_dynamic_model_right() — MPC right")
        amp = list(ampli[0])
        dur = list(duri[0])
        self._send_segment_array(
            amp_array_uv=amp,
            dur_array_us=dur,
            target_unit_ids={self.right_unit_id},
            user_id=stim_pulse.EVENT_USER_ID_MPC_RIGHT,
            label_prefix="mpc_right",
        )

    # ---------- 兜底默认惩罚 ----------

    def _send_default_punish(self, target_unit_id, user_id, label_prefix):
        """
        set_sti_signal 未被调用时的兜底惩罚刺激：5 Hz 单脉冲序列，1 个 100 ms 段。
        语义上等价于 MCS 端在 stim_setting 没配置时的退化路径。
        """
        self._send_pulse_train_freq(
            freq_hz=DEFAULT_PUNISH_FREQ_HZ,
            amplitude_uv=DEFAULT_PUNISH_AMPLITUDE_UV,
            phase_us=DEFAULT_PUNISH_PHASE_US,
            target_unit_ids={target_unit_id},
            user_id=user_id,
            label_prefix=label_prefix,
            polarity="positive_first",
        )


# ---------- 私有工具 ----------

def _make_burst_mask(n_slots, duty, min_len, rng):
    """
    完整照搬 MCS set_sti_signal.make_burst_mask（不修改业务逻辑）：
      生成长度 n_slots 的 0/1 mask，1 表示该 slot 打刺激；
      1 以 burst（连续 1）形式分布，每 burst ≥ min_len，burst 起点随机。
    """
    if n_slots <= 0:
        return []
    n_on = int(round(n_slots * duty))
    n_on = max(0, min(n_on, n_slots))
    n_off = n_slots - n_on
    if n_on == 0:
        return [0] * n_slots
    if n_on == n_slots:
        return [1] * n_slots
    if n_on < min_len:
        return [1] * n_on + [0] * (n_slots - n_on)

    max_b = min(n_on // min_len, n_off + 1)
    b = rng.randint(1, max_b)

    burst_lens = [min_len] * b
    remaining = n_on - min_len * b
    for _ in range(remaining):
        burst_lens[rng.randrange(b)] += 1

    internal_min = b - 1
    zeros_left = n_off - internal_min

    gaps = [0] * (b + 1)
    for gi in range(1, b):
        gaps[gi] = 1
    for _ in range(zeros_left):
        gaps[rng.randrange(b + 1)] += 1

    mask = []
    mask.extend([0] * gaps[0])
    for idx, L in enumerate(burst_lens):
        mask.extend([1] * L)
        mask.extend([0] * gaps[idx + 1])
    return mask[:n_slots]
