"""
Maxwell connect / power_up 单独 send 语义证伪脚本（方案 C，扩展版）。

背景
----
平台 stim_pool.activate / deactivate 用 `mx.send(StimulationUnit(u).connect(T/F))`
单独 send 切换 unit 输出。lr_smoke 实测发现 unit 切换后刺激总挂在某一通道，
怀疑 connect(False) 单独 send 不生效。Maxwell 官方 stimulate.html 的运行时
切换模板用整套链式 send 启用 + power_up(False) 单独 send 关闭，从不用
connect(False) 单独 send。

代码层面证据采集
----------------
API 不直接暴露 StimulationUnit.connect / power_up 当前状态查询，但有以下
routing 层 / readout 层 query 可侧面探测，并都返回字符串可对比：

  array.query_stimulation_at_electrode(ele)    返回 'OK'/'ERROR' 或 unit_id 字符串
  array.query_amplifier_at_electrode(ele)      返回 amp 通道
  array.query_amplifier_at_stimulation(unit)   返回 unit 对应的 amp 通道
  StimulationUnit(unit).get_readout_channel()  返回 unit 自身 readout 通道

每个阶段的关键动作前后都做 probe，加上每次 mx.send 的真实返回值，能拼出
"哪一次单独 send 是否真的改变了某些可观察状态" 的指纹。再结合 MaxLab Live
录制的波形，三方交叉印证。

阶段
----
P0: 启动后 baseline probe（stim_pool 已 power_up(True)+connect(False)）
P1: unit_left 整套链式 power_up(T).connect(T).voltage_mode.dac_source(0)
P2: unit_left 单独 send connect(False)            ← 关键判断 1
P3: unit_right 整套链式 power_up(T).connect(T).voltage_mode.dac_source(0)
P4: unit_right 单独 send power_up(False)          ← 关键判断 2
P5: unit_left 整套链式 connect=False（power_up 仍 True）← 与 P2 对照
P6: unit_left 整套链式 power_up=False                  ← 与 P4 对照

每个阶段顺序：probe before → 动作 mx.send → 打印返回值 → probe after →
发 5 脉冲 sequence → probe after-send → gap。

使用
----
    /home/maxwell/metaboc-env/bin/python tools/maxwell_connect_probe.py \\
        /path/to/your.cfg --left 2661 --right 18884

  - --gap 3.0   段间隔秒数（默认 3.0）

刺激参数：5 pulses / 100Hz / 200mV / 每相 200μs。Event user_id 命名空间 11-16
区分 6 个阶段，便于事后从 frame metadata 与 MaxLab Live 录制波形定位。
"""

import argparse
import sys
import time
from pathlib import Path


DAC_CENTER = 512
AMPLITUDE_UV = 200_000
PHASE_US = 200
PERIOD_US = 10_000
N_PULSES = 5
SAMPLE_DURATION_US = 50

USER_ID_P1 = 11
USER_ID_P2 = 12
USER_ID_P3 = 13
USER_ID_P4 = 14
USER_ID_P5 = 15
USER_ID_P6 = 16


def _print_banner(title):
    print()
    print("=" * 72)
    print("  " + title)
    print("=" * 72)


def _build_pulse_train(dac_lsb_mv, label_prefix, user_id):
    import maxlab as mx

    amp_dac = int(round((AMPLITUDE_UV / 1000.0) / dac_lsb_mv))
    phase_samples = max(1, int(round(PHASE_US / SAMPLE_DURATION_US)))
    gap_samples = max(1, int(round((PERIOD_US - 2 * PHASE_US) / SAMPLE_DURATION_US)))

    seq = mx.Sequence(initial_delay=100)
    for i in range(N_PULSES):
        seq.append(mx.Event(0, 1, user_id, "{} pulse_{}".format(label_prefix, i + 1)))
        seq.append(mx.DAC(0, DAC_CENTER - amp_dac))
        seq.append(mx.DelaySamples(phase_samples))
        seq.append(mx.DAC(0, DAC_CENTER + amp_dac))
        seq.append(mx.DelaySamples(phase_samples))
        seq.append(mx.DAC(0, DAC_CENTER))
        seq.append(mx.DelaySamples(gap_samples))
    seq.append(mx.DAC(0, DAC_CENTER))
    return seq


def _safe_call(label, fn):
    """容错调用并打印 repr 返回值或异常。"""
    try:
        rv = fn()
        print("    {:<55s} -> {!r}".format(label, rv))
        return rv
    except Exception as e:
        print("    {:<55s} !! raised {!r}".format(label, e))
        return None


def _probe(tag, array, left_ele, right_ele, left_unit, right_unit):
    """探测当前 routing / readout 层可观察状态。"""
    import maxlab as mx
    print()
    print("  ---- probe [{}] ----".format(tag))

    _safe_call(
        "array.query_stimulation_at_electrode(left={})".format(left_ele),
        lambda: array.query_stimulation_at_electrode(left_ele),
    )
    _safe_call(
        "array.query_stimulation_at_electrode(right={})".format(right_ele),
        lambda: array.query_stimulation_at_electrode(right_ele),
    )
    _safe_call(
        "array.query_amplifier_at_electrode(left={})".format(left_ele),
        lambda: array.query_amplifier_at_electrode(left_ele),
    )
    _safe_call(
        "array.query_amplifier_at_electrode(right={})".format(right_ele),
        lambda: array.query_amplifier_at_electrode(right_ele),
    )
    _safe_call(
        "array.query_amplifier_at_stimulation(unit_left={})".format(left_unit),
        lambda: array.query_amplifier_at_stimulation(left_unit),
    )
    _safe_call(
        "array.query_amplifier_at_stimulation(unit_right={})".format(right_unit),
        lambda: array.query_amplifier_at_stimulation(right_unit),
    )
    _safe_call(
        "StimulationUnit({}).get_readout_channel()".format(left_unit),
        lambda: mx.StimulationUnit(left_unit).get_readout_channel(),
    )
    _safe_call(
        "StimulationUnit({}).get_readout_channel()".format(right_unit),
        lambda: mx.StimulationUnit(right_unit).get_readout_channel(),
    )


def _send_and_print(label, cmd_factory):
    """mx.send + 打印真实返回值。cmd_factory 是无参 lambda 返回 cmd 对象。"""
    import maxlab as mx
    print()
    print("  >> mx.send: {}".format(label))
    cmd = cmd_factory()
    rv = mx.send(cmd)
    print("     return = {!r}".format(rv))
    return rv


def main():
    parser = argparse.ArgumentParser(
        description="Maxwell connect/power_up single-send semantics probe (extended)."
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

    _print_banner("[1/8] start_session")
    sys_obj.start_session(
        cfg_path=str(cfg_path),
        record_electrodes=None,
        stim_electrodes=[args.left, args.right],
        role_mapping=None,
    )
    stm = sys_obj.stimulating
    pool = sys_obj.stim_pool
    array = pool._array
    left_unit = stm.left_unit_id
    right_unit = stm.right_unit_id
    dac_lsb_mv = stm._dac_lsb_mv
    amp_dac = int(round((AMPLITUDE_UV / 1000.0) / dac_lsb_mv))

    print("  left  electrode={} unit={}".format(args.left, left_unit))
    print("  right electrode={} unit={}".format(args.right, right_unit))
    print("  DAC LSB = {:.4f} mV/bit".format(dac_lsb_mv))
    print("  amp_dac = {} bits (electrode-side amplitude = {} mV)".format(
        amp_dac, AMPLITUDE_UV / 1000
    ))

    seq_p1 = _build_pulse_train(dac_lsb_mv, "P1_left_chain_ON", USER_ID_P1)
    seq_p2 = _build_pulse_train(dac_lsb_mv, "P2_left_after_single_connect_False", USER_ID_P2)
    seq_p3 = _build_pulse_train(dac_lsb_mv, "P3_right_chain_ON", USER_ID_P3)
    seq_p4 = _build_pulse_train(dac_lsb_mv, "P4_right_after_single_power_up_False", USER_ID_P4)
    seq_p5 = _build_pulse_train(dac_lsb_mv, "P5_left_chain_connect_False", USER_ID_P5)
    seq_p6 = _build_pulse_train(dac_lsb_mv, "P6_left_chain_power_up_False", USER_ID_P6)

    try:
        _print_banner("[2/8] P0 — baseline probe (启动后 stim_pool 已 power_up(T)+connect(F))")
        _probe("P0_baseline", array, args.left, args.right, left_unit, right_unit)
        time.sleep(args.gap)

        # ============ P1 ============
        _print_banner("[3/8] P1 — unit_left 整套链式 send: power_up(T).connect(T).voltage_mode.dac_source(0)")
        _probe("P1_before", array, args.left, args.right, left_unit, right_unit)
        _send_and_print(
            "unit_left FULL chain (power_up=T,connect=T,voltage,dac0)",
            lambda: (mx.StimulationUnit(left_unit)
                     .power_up(True).connect(True)
                     .set_voltage_mode().dac_source(0)),
        )
        _probe("P1_after_send", array, args.left, args.right, left_unit, right_unit)
        time.sleep(0.5)
        print("\n  >> seq.send() P1_left_chain_ON (label user_id 11)")
        print("  >> 预期 electrode_left ({}) 上有 5 个 200mV 双相脉冲".format(args.left))
        seq_p1.send()
        time.sleep(args.gap)

        # ============ P2 ============
        _print_banner("[4/8] P2 — unit_left 单独 send connect(False)")
        _probe("P2_before", array, args.left, args.right, left_unit, right_unit)
        _send_and_print(
            "unit_left SINGLE connect(False)",
            lambda: mx.StimulationUnit(left_unit).connect(False),
        )
        _probe("P2_after_send", array, args.left, args.right, left_unit, right_unit)
        time.sleep(0.5)
        print("\n  >> seq.send() P2_left_after_single_connect_False (label user_id 12)")
        print("  *** 关键判断 1: electrode_left 上是否还有波形? ***")
        seq_p2.send()
        time.sleep(args.gap)

        # ============ P3 ============
        _print_banner("[5/8] P3 — unit_right 整套链式 send (与 P1 同样模板)")
        _probe("P3_before", array, args.left, args.right, left_unit, right_unit)
        _send_and_print(
            "unit_right FULL chain",
            lambda: (mx.StimulationUnit(right_unit)
                     .power_up(True).connect(True)
                     .set_voltage_mode().dac_source(0)),
        )
        _probe("P3_after_send", array, args.left, args.right, left_unit, right_unit)
        time.sleep(0.5)
        print("\n  >> seq.send() P3_right_chain_ON (label user_id 13)")
        print("  >> 预期 electrode_right ({}) 有 5 脉冲".format(args.right))
        print("  >> 附加观察 electrode_left ({}) 是否同时也有波形:".format(args.left))
        print("       有 → unit_left 仍 connect=True（P2 未生效）+ DAC0 共享驱动")
        print("       无 → unit_left 已 disconnect")
        seq_p3.send()
        time.sleep(args.gap)

        # ============ P4 ============
        _print_banner("[6/8] P4 — unit_right 单独 send power_up(False)")
        _probe("P4_before", array, args.left, args.right, left_unit, right_unit)
        _send_and_print(
            "unit_right SINGLE power_up(False)",
            lambda: mx.StimulationUnit(right_unit).power_up(False),
        )
        _probe("P4_after_send", array, args.left, args.right, left_unit, right_unit)
        time.sleep(0.5)
        print("\n  >> seq.send() P4_right_after_single_power_up_False (label user_id 14)")
        print("  *** 关键判断 2: electrode_right 上是否还有波形? ***")
        seq_p4.send()
        time.sleep(args.gap)

        # ============ P5 ============
        _print_banner("[7/8] P5 — unit_left 整套链式 connect=False (power_up 仍 True；与 P2 单独 send 对照)")
        _probe("P5_before", array, args.left, args.right, left_unit, right_unit)
        _send_and_print(
            "unit_left FULL chain (power_up=T,connect=F,voltage,dac0)",
            lambda: (mx.StimulationUnit(left_unit)
                     .power_up(True).connect(False)
                     .set_voltage_mode().dac_source(0)),
        )
        _probe("P5_after_send", array, args.left, args.right, left_unit, right_unit)
        time.sleep(0.5)
        print("\n  >> seq.send() P5_left_chain_connect_False (label user_id 15)")
        print("  *** electrode_left 上是否有波形?")
        print("      若 P2 未生效但 P5 生效 → 整套链式 send 比单独 send 强；走方案 B ***")
        seq_p5.send()
        time.sleep(args.gap)

        # ============ P6 ============
        _print_banner("[8/8] P6 — unit_left 整套链式 power_up=False (与 P4 单独 send 对照)")
        _probe("P6_before", array, args.left, args.right, left_unit, right_unit)
        _send_and_print(
            "unit_left FULL chain power_up=False",
            lambda: (mx.StimulationUnit(left_unit)
                     .power_up(False).connect(False)
                     .set_voltage_mode().dac_source(0)),
        )
        _probe("P6_after_send", array, args.left, args.right, left_unit, right_unit)
        time.sleep(0.5)
        print("\n  >> seq.send() P6_left_chain_power_up_False (label user_id 16)")
        print("  *** electrode_left 上是否有波形? 整套链式 power_up=False 是否生效")
        seq_p6.send()
        time.sleep(args.gap)

    finally:
        _print_banner("cleanup — stop_connect")
        try:
            sys_obj.stop_connect()
            print("  stop_connect OK")
        except Exception as exc:
            print("  stop_connect raised: {!r}".format(exc))

    _print_banner("Event labels 速查（事后看 frame metadata）")
    print("  user_id 11 = P1_left_chain_ON_pulse_*               (5 个，期望 left 有)")
    print("  user_id 12 = P2_left_after_single_connect_False_*   (5 个，关键判断 1)")
    print("  user_id 13 = P3_right_chain_ON_pulse_*              (5 个，期望 right 有 + 看 left 是否串)")
    print("  user_id 14 = P4_right_after_single_power_up_False_* (5 个，关键判断 2)")
    print("  user_id 15 = P5_left_chain_connect_False_pulse_*    (5 个，整套链式 connect=False 对照)")
    print("  user_id 16 = P6_left_chain_power_up_False_pulse_*   (5 个，整套链式 power_up=False 对照)")


if __name__ == "__main__":
    main()
