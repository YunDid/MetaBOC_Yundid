"""
Maxwell connect / power_up 单独 send 语义证伪脚本（方案 C）。

背景
----
平台 stim_pool.activate / deactivate 用 `mx.send(StimulationUnit(u).connect(T/F))`
单独 send 切换 unit 输出。lr_smoke 实测发现 unit 切换后刺激总挂在某一通道，
怀疑 connect(False) 单独 send 不生效。Maxwell 官方 stimulate.html 的运行时
切换模板用整套链式 send 启用 + power_up(False) 单独 send 关闭，从不用
connect(False) 单独 send。

本脚本顺序发 4 段 sequence，每段插 mx.Event 区分阶段，结合 MaxLab Live 同步
录制即可定位 connect / power_up 单独 send 的真实语义。

阶段
----
P1: unit_left 整套链式 power_up(True).connect(True).voltage_mode.dac_source(0)
    → 发 5 个脉冲；预期 electrode_left 有波形
P2: unit_left 单独 send connect(False)
    → 发 5 个脉冲；electrode_left 若仍有波形 → connect(False) 单独 send 无效
P3: unit_right 整套链式 power_up(True).connect(True).voltage_mode.dac_source(0)
    → 发 5 个脉冲；预期 electrode_right 有波形
       附加观察：electrode_left 此时是否同时有波形（dac_source(0) 共享 + P2
       是否生效的双重检验）
P4: unit_right 单独 send power_up(False)
    → 发 5 个脉冲；electrode_right 若仍有波形 → power_up(False) 单独 send 也无效

使用
----
    /home/maxwell/metaboc-env/bin/python tools/maxwell_connect_probe.py \\
        /path/to/your.cfg --left 2661 --right 18884

  - --gap 3.0   段间隔秒数（默认 3.0；放宽便于 MaxLab Live 录制肉眼区分）

每段刺激参数：5 pulses / 100Hz / 200mV / 每相 200μs（与 lr_smoke env 一致以便对照）。
"""

import argparse
import sys
import time
from pathlib import Path


DAC_CENTER = 512
AMPLITUDE_UV = 200_000      # 200 mV
PHASE_US = 200              # 每相 200 μs
PERIOD_US = 10_000          # 100 Hz
N_PULSES = 5
SAMPLE_DURATION_US = 50

# Event user_id 命名空间（避开业务接口 1-9）
USER_ID_P1 = 11
USER_ID_P2 = 12
USER_ID_P3 = 13
USER_ID_P4 = 14


def _print_banner(title):
    print()
    print("=" * 64)
    print("  " + title)
    print("=" * 64)


def _build_pulse_train(dac_lsb_mv, label_prefix, user_id):
    """构造 5 脉冲、双相 200mV / 200μs / 100Hz 序列。

    电压模式下放大器反相：DAC bit < 512 → 电极端 +V；DAC bit > 512 → 电极端 -V。
    本序列电极端波形：正相先 [+200mV, -200mV, 0]。
    """
    import maxlab as mx

    amp_dac = int(round((AMPLITUDE_UV / 1000.0) / dac_lsb_mv))
    phase_samples = max(1, int(round(PHASE_US / SAMPLE_DURATION_US)))
    gap_samples = max(1, int(round((PERIOD_US - 2 * PHASE_US) / SAMPLE_DURATION_US)))

    seq = mx.Sequence(initial_delay=100)
    for i in range(N_PULSES):
        seq.append(mx.Event(0, 1, user_id, "{} pulse_{}".format(label_prefix, i + 1)))
        seq.append(mx.DAC(0, DAC_CENTER - amp_dac))     # 电极端 +200mV
        seq.append(mx.DelaySamples(phase_samples))
        seq.append(mx.DAC(0, DAC_CENTER + amp_dac))     # 电极端 -200mV
        seq.append(mx.DelaySamples(phase_samples))
        seq.append(mx.DAC(0, DAC_CENTER))               # 电极端 0
        seq.append(mx.DelaySamples(gap_samples))
    seq.append(mx.DAC(0, DAC_CENTER))                   # 收尾归零
    return seq


def main():
    parser = argparse.ArgumentParser(
        description="Maxwell connect/power_up single-send semantics probe."
    )
    parser.add_argument("cfg_path", help="MaxLab Live exported .cfg path")
    parser.add_argument("--left", type=int, required=True,
                        help="Left wheel stim electrode ID")
    parser.add_argument("--right", type=int, required=True,
                        help="Right wheel stim electrode ID")
    parser.add_argument("--gap", type=float, default=3.0,
                        help="Inter-phase gap seconds (default 3.0)")
    args = parser.parse_args()

    cfg_path = Path(args.cfg_path)
    if not cfg_path.exists():
        print("cfg not found: {}".format(cfg_path))
        sys.exit(2)
    if args.left == args.right:
        print("--left and --right must differ")
        sys.exit(2)

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    print("cfg_path  = {}".format(cfg_path))
    print("left ele  = {}".format(args.left))
    print("right ele = {}".format(args.right))
    print("gap       = {:.1f}s".format(args.gap))

    from src.system_device.maxwell_sys import MaxwellSystem
    import maxlab as mx

    sys_obj = MaxwellSystem()

    _print_banner("[1/6] start_session (启动后 stim_pool 已 power_up(True)+connect(False))")
    sys_obj.start_session(
        cfg_path=str(cfg_path),
        record_electrodes=None,
        stim_electrodes=[args.left, args.right],
        role_mapping=None,
    )
    stm = sys_obj.stimulating
    left_unit = stm.left_unit_id
    right_unit = stm.right_unit_id
    dac_lsb_mv = stm._dac_lsb_mv

    print("  left  electrode={} unit={}".format(args.left, left_unit))
    print("  right electrode={} unit={}".format(args.right, right_unit))
    print("  DAC LSB = {:.4f} mV/bit".format(dac_lsb_mv))

    seq_p1 = _build_pulse_train(dac_lsb_mv, "P1_left_ON", USER_ID_P1)
    seq_p2 = _build_pulse_train(dac_lsb_mv, "P2_left_after_connect_False", USER_ID_P2)
    seq_p3 = _build_pulse_train(dac_lsb_mv, "P3_right_ON", USER_ID_P3)
    seq_p4 = _build_pulse_train(dac_lsb_mv, "P4_right_after_power_up_False", USER_ID_P4)

    try:
        _print_banner("[2/6] P1 — unit_left 整套链式 send 启用")
        cmd = (mx.StimulationUnit(left_unit)
               .power_up(True).connect(True)
               .set_voltage_mode().dac_source(0))
        rc = mx.send(cmd)
        print("  mx.send(unit_left full-chain) -> {!r}".format(rc))
        time.sleep(0.5)
        print("  发 P1 sequence (5 pulses, label P1_left_ON)")
        print("  >> 预期 electrode_left ({}) 上有 5 个 200mV 双相脉冲".format(args.left))
        seq_p1.send()
        time.sleep(args.gap)

        _print_banner("[3/6] P2 — unit_left 单独 send connect(False)")
        rc = mx.send(mx.StimulationUnit(left_unit).connect(False))
        print("  mx.send(unit_left.connect(False) single) -> {!r}".format(rc))
        time.sleep(0.5)
        print("  发 P2 sequence (5 pulses, label P2_left_after_connect_False)")
        print("  *** 关键判断 1: electrode_left ({}) 上是否还有波形? ***".format(args.left))
        print("      仍有 → connect(False) 单独 send 不生效（证实方案 A 必要）")
        print("      已无 → connect(False) 单独 send 生效（lr_smoke 故障另有原因）")
        seq_p2.send()
        time.sleep(args.gap)

        _print_banner("[4/6] P3 — unit_right 整套链式 send 启用")
        cmd = (mx.StimulationUnit(right_unit)
               .power_up(True).connect(True)
               .set_voltage_mode().dac_source(0))
        rc = mx.send(cmd)
        print("  mx.send(unit_right full-chain) -> {!r}".format(rc))
        time.sleep(0.5)
        print("  发 P3 sequence (5 pulses, label P3_right_ON)")
        print("  >> 预期 electrode_right ({}) 上有 5 个 200mV 双相脉冲".format(args.right))
        print("  >> 附加观察 electrode_left ({}):".format(args.left))
        print("      也有波形 → unit_left 仍 connect=True（P2 未生效）+ DAC0 共享驱动两 unit")
        print("      无波形   → unit_left 已 disconnect")
        seq_p3.send()
        time.sleep(args.gap)

        _print_banner("[5/6] P4 — unit_right 单独 send power_up(False)")
        rc = mx.send(mx.StimulationUnit(right_unit).power_up(False))
        print("  mx.send(unit_right.power_up(False) single) -> {!r}".format(rc))
        time.sleep(0.5)
        print("  发 P4 sequence (5 pulses, label P4_right_after_power_up_False)")
        print("  *** 关键判断 2: electrode_right ({}) 上是否还有波形? ***".format(args.right))
        print("      仍有 → power_up(False) 单独 send 也不生效（深层固件问题）")
        print("      已无 → power_up(False) 单独 send 生效（方案 A 可行）")
        seq_p4.send()
        time.sleep(args.gap)

    finally:
        _print_banner("[6/6] stop_connect")
        try:
            sys_obj.stop_connect()
            print("  stop_connect OK")
        except Exception as exc:
            print("  stop_connect raised: {!r}".format(exc))

    _print_banner("决策表（结合 MaxLab Live 录制波形 + frame metadata Event labels）")
    print("""
  Event labels（用 user_id 区分阶段）：
    user_id 11 = P1_left_ON_pulse_*           (5 个)
    user_id 12 = P2_left_after_connect_False_pulse_*  (5 个)
    user_id 13 = P3_right_ON_pulse_*          (5 个)
    user_id 14 = P4_right_after_power_up_False_pulse_*(5 个)

  关键判断 1 — connect(False) 单独 send 是否有效：
    P1 left 有 / P2 left 无                  → connect(False) 有效
    P1 left 有 / P2 left 仍有                → connect(False) 无效（必须走方案 A）

  关键判断 2 — power_up(False) 单独 send 是否有效：
    P3 right 有 / P4 right 无                → power_up(False) 有效（方案 A 可行）
    P3 right 有 / P4 right 仍有              → power_up(False) 也无效（深层问题，需重谋）

  附加观察 — DAC0 共享驱动多 unit 是否生效：
    P3 时 left 也有波形 → unit_left 未 disconnect + DAC0 同步驱动两 unit
    P3 时 left 无波形   → unit_left 已 disconnect
""")


if __name__ == "__main__":
    main()
