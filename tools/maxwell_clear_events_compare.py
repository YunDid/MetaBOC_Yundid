"""
Maxwell mx.clear_events 频次对照测试。

目的
----
验证"每次 seq.send 前都 mx.clear_events 会擦掉之前的 mx.Event 标记"
这一假设。两种调用策略下事后 .h5 frame 元数据中刺激事件总数会有量化差异。

策略对照：
  - mode every : 每次 build sequence 后、seq.send 前 clear（旧实现行为）
                 → 仅最后 1 round 的事件保留，前面所有 round 标记被擦
  - mode once  : 启动期 attach_stim_pool 已 clear 过一次，运行时不再 clear
                 → 所有 round 的事件全保留（新实现行为，commit 475b31f）

不修改 stimulation_maxwell.py 等生产代码。复用 MaxwellSystem 启动 +
stim_pool，运行循环里自己手动构造 mx.Sequence 并控制 clear_events 调用。

使用
----
两个模式各跑一次，每次都用 MaxLab Live 录 .h5（不录就只能看 stdout 推断）：

  # mode every（模拟旧实现）
  /home/maxwell/metaboc-env/bin/python tools/maxwell_clear_events_compare.py \\
      /home/maxwell/configs/260227/pi_16h44m18s.cfg \\
      --left 2661 --right 18884 \\
      --n-rounds 3 --n-pulses 5 --clear-mode every

  # mode once（验证当前实现）
  ... --clear-mode once

事后用 MaxLab Live 打开两份 .h5，看 frame metadata Event 数：
  - every: 应只见 round_3/pulse_1..5（共 5 个），round_1/* 与 round_2/* 全失踪
  - once : 应见 round_1/* 到 round_3/* 全部（共 15 个）

如果 every 模式 .h5 仍能看到 round_1 / round_2 → 假设错了，clear_events
不影响已写入的 frame 元数据；如果只见 round_3 → 假设成立，旧实现确实
擦掉了刺激历史。

注意
----
- 测试只发到左轮电极（--left），右轮 unit 在循环开始前 deactivate
- amp 默认 200 mV / 200 us 双相 / 100 ms 间隔（与 maxwell_stim_smoke.py 一致）
- inter-round gap 默认 2 秒（让 sequence 发完且波形可分辨）
"""

import argparse
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Maxwell mx.clear_events frequency comparison smoke."
    )
    parser.add_argument("cfg_path", help="MaxLab Live exported .cfg path")
    parser.add_argument("--left", type=int, required=True,
                        help="Left wheel stim electrode (target of all pulses)")
    parser.add_argument("--right", type=int, required=True,
                        help="Right wheel stim electrode (deactivated, kept in pool)")
    parser.add_argument("--n-rounds", type=int, default=3,
                        help="Number of seq.send rounds")
    parser.add_argument("--n-pulses", type=int, default=5,
                        help="Pulses per round (each tagged with mx.Event)")
    parser.add_argument("--clear-mode", choices=["every", "once"], required=True,
                        help="every = clear before each seq.send (old behavior); "
                             "once = no clear in loop (current behavior)")
    parser.add_argument("--amplitude-mv", type=int, default=200,
                        help="Per-phase amplitude in mV (default 200)")
    parser.add_argument("--phase-samples", type=int, default=4,
                        help="Samples per phase (default 4 = 200us)")
    parser.add_argument("--inter-pulse-samples", type=int, default=2000,
                        help="Samples between pulses (default 2000 = 100ms)")
    parser.add_argument("--gap", type=float, default=2.0,
                        help="Inter-round gap seconds (default 2.0)")
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

    print("=" * 60)
    print("clear-mode    = {}".format(args.clear_mode))
    print("n_rounds      = {}".format(args.n_rounds))
    print("n_pulses      = {}".format(args.n_pulses))
    print("target left   = {}".format(args.left))
    print("idle  right   = {}".format(args.right))
    total_events = args.n_rounds * args.n_pulses
    print()
    print("If clear_events frequency assumption is correct:")
    if args.clear_mode == "every":
        print("  expected Event count in .h5 = {} (only last round survives;".format(args.n_pulses))
        print("                                   {} events from rounds 1..{} wiped)".format(
            (args.n_rounds - 1) * args.n_pulses, args.n_rounds - 1
        ))
    else:
        print("  expected Event count in .h5 = {} (all {} rounds preserved)".format(
            total_events, args.n_rounds
        ))
    print("=" * 60)

    from src.system_device.maxwell_sys import MaxwellSystem
    import maxlab as mx

    sys_obj = MaxwellSystem()
    failed = False

    try:
        # 完整启动序列；attach_stim_pool 内部已做一次 mx.clear_events
        sys_obj.start_session(
            cfg_path=str(cfg_path),
            record_electrodes=None,
            stim_electrodes=[args.left, args.right],
            role_mapping=None,
        )

        stm = sys_obj.stimulating
        target_unit = stm.left_unit_id
        target_electrode = stm.left_electrode
        idle_electrode = stm.right_electrode

        # 把 right wheel 的 unit 关掉，让本次发放只驱动 left wheel
        # （attach_stim_pool 后两个 unit 都是 connect=True；deactivate right
        #  让 DAC0 序列只送到 left unit）
        sys_obj.stim_pool.deactivate([idle_electrode])
        print("[setup] deactivated right wheel electrode={} (unit={}); "
              "left target electrode={} (unit={})".format(
                  idle_electrode, stm.right_unit_id,
                  target_electrode, target_unit,
              ))

        dac_lsb = float(mx.query_DAC_lsb_mV())
        amp_dac = int(args.amplitude_mv / dac_lsb)
        print("[setup] DAC LSB = {:.4f} mV/bit, amp_dac = {} bits".format(dac_lsb, amp_dac))
        print()

        # 主循环：每 round 构造一个 sequence + 视 mode 决定 clear
        for r in range(args.n_rounds):
            print("--- round {}/{} ---".format(r + 1, args.n_rounds))

            seq = mx.Sequence()
            for i in range(args.n_pulses):
                # user_id 在整个测试中唯一编号；label 含 round/pulse 便于事后过滤
                user_id = r * args.n_pulses + i + 1
                label = "round_{} pulse_{} amp_{}mV".format(
                    r + 1, i + 1, args.amplitude_mv
                )
                seq.append(mx.Event(0, 1, user_id, label))
                # 双相 DAC（电压模式反相，先减后加 → 电极端正相先）
                seq.append(mx.DAC(0, 512 - amp_dac))
                seq.append(mx.DelaySamples(args.phase_samples))
                seq.append(mx.DAC(0, 512 + amp_dac))
                seq.append(mx.DelaySamples(args.phase_samples))
                seq.append(mx.DAC(0, 512))
                seq.append(mx.DelaySamples(args.inter_pulse_samples))

            if args.clear_mode == "every":
                print("  mx.clear_events() <-- WIPES all prior Event marks")
                mx.clear_events()

            print("  seq.send() ({} pulses, user_id range {}..{})".format(
                args.n_pulses,
                r * args.n_pulses + 1,
                r * args.n_pulses + args.n_pulses,
            ))
            seq.send()

            # 等 sequence 发完
            total_samples = args.n_pulses * (2 * args.phase_samples + args.inter_pulse_samples)
            wait_s = total_samples * 50e-6 + 0.3
            time.sleep(wait_s)
            print("  ... waited {:.2f}s for sequence to complete".format(wait_s))

            if r < args.n_rounds - 1:
                time.sleep(args.gap)
                print("  ... inter-round gap {:.1f}s".format(args.gap))

        print()
        print("=" * 60)
        print("ALL {} ROUNDS DISPATCHED.".format(args.n_rounds))
        print("=" * 60)
        if args.clear_mode == "every":
            print("Open the recorded .h5 and check Event labels:")
            print("  IF only 'round_{}/pulse_*' is visible (n={}) → assumption confirmed:".format(
                args.n_rounds, args.n_pulses
            ))
            print("     clear_events wipes the frame-metadata event buffer")
            print("  IF earlier rounds also visible → assumption wrong:")
            print("     clear_events doesn't touch already-recorded frames")
        else:
            print("Open the recorded .h5 and check Event labels:")
            print("  expect round_1/pulse_1..{} through round_{}/pulse_1..{}".format(
                args.n_pulses, args.n_rounds, args.n_pulses
            ))
            print("  total = {} Event marks".format(total_events))

    except Exception:
        failed = True
        import traceback
        print()
        print("SMOKE FAILED — traceback below:")
        traceback.print_exc()

    finally:
        try:
            sys_obj.stop_connect()
            print("[cleanup] stop_connect OK")
        except Exception as exc:
            print("[cleanup] stop_connect raised: {!r}".format(exc))

    print()
    if failed:
        print("RESULT: FAIL")
        sys.exit(1)
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
