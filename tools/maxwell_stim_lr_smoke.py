"""
Maxwell 左右轮 stim 业务接口冒烟测试（Phase B2 真发刺激路径）。

补 maxwell_phase_b_smoke.py（只验骨架，不发刺激）与 maxwell_stim_smoke.py
（只发原始 DAC sequence，绕过业务层）之间的空隙：

  - 复用 MaxwellSystem.start_session 完成完整启动 + left/right wheel 角色绑定
  - 直接调 StimulationMaxwell 各 update_* 接口逐项验证
  - 每段间 sleep 等发完，避免 sequence 重叠
  - 打印目标电极 / 单元 / 期望 Event 标签，便于 MaxLab Live 录制对照

为什么需要本脚本
----------------
Phase B 阶段 RecordingMaxwell.get_recording() 是 stub 占位（[], []），C++
binary DataStreamer 在 Phase D。没 spike 输入 → encode 不出频率 → 主路径
update_record_stimulation 不会被 main.py 自动触发。GUI 端到端只能验
菜单切换 + cfg 选择 + 角色绑定到日志，无法触发刺激接口走真发。

使用
----
Linux 真机（mxwserver 运行中）：

    /home/maxwell/metaboc-env/bin/python tools/maxwell_stim_lr_smoke.py \\
        /path/to/your.cfg --left 2661 --right 18884

  - --tests env_left,env_right,punish_left,...   选择要跑的测试集（默认全跑）
  - --gap 2.0                                    inter-test gap seconds

成功标志
--------
  - 终端逐项 fired in N.Ns；active electrodes after = ...
  - mxwserver 端无 'Error with' 行
  - MaxLab Live 同步录制时，frame metadata 可见各 Event label

刺激不串扰检查
--------------
每个 env_* / punish_* / reward_left / reward_right / mpc_* 测试发完后，
脚本打印 active electrodes — 应当只剩目标侧的电极（共享 DAC0 时另一侧
unit 必须 connect=False）。reward_both 测试发完后，应见两侧都 active。
"""

import argparse
import sys
import time
import traceback
from pathlib import Path


def _print_banner(title):
    print()
    print("=" * 60)
    print("  " + title)
    print("=" * 60)


def _wait(seconds, why=""):
    print("  ... wait {:.1f}s ({})".format(seconds, why))
    time.sleep(seconds)


# 测试注册表：name -> (description, callable(stim) -> None, post_sleep_seconds)
# post_sleep_seconds 选 1.5s 是因为单次 update_* 最长序列大约 100ms（10 脉冲 ×
# 10ms 周期）+ Sequence(initial_delay=100) 的 5ms，余量充分。
TEST_REGISTRY = {
    "env_left": (
        "env: left>=right -> left wheel fires (20Hz, ~2 pulses)",
        lambda stm: stm.update_record_stimulation(left=20, right=10),
        1.5,
        "left",
    ),
    "env_right": (
        "env: right>left -> right wheel fires (20Hz, ~2 pulses)",
        lambda stm: stm.update_record_stimulation(left=10, right=20),
        1.5,
        "right",
    ),
    "punish_left": (
        "punish: left collision (default fallback waveform, set_sti_signal not called)",
        lambda stm: stm.update_stimulation_left(),
        1.5,
        "left",
    ),
    "punish_right": (
        "punish: right collision (default fallback waveform)",
        lambda stm: stm.update_stimulation_right(),
        1.5,
        "right",
    ),
    "reward_left": (
        "reward: left only (100Hz/75mV/100ms biphasic, 10 pulses)",
        lambda stm: stm.update_stimulation_left_reward(),
        1.5,
        "left",
    ),
    "reward_right": (
        "reward: right only",
        lambda stm: stm.update_stimulation_right_reward(),
        1.5,
        "right",
    ),
    "reward_both": (
        "reward: both wheels simultaneously (shared DAC0 drives both units)",
        lambda stm: stm.update_stimulation_left_right_reward(),
        1.5,
        "both",
    ),
    "mpc_left": (
        "mpc: left dynamic-model array path (single biphasic [+75mV,-75mV,0])",
        lambda stm: stm.update_record_stimulation_dynamic_model_left(
            ampli=[[75000, -75000, 0]],
            duri=[[200, 200, 9600]],
        ),
        1.5,
        "left",
    ),
    "mpc_right": (
        "mpc: right dynamic-model array path",
        lambda stm: stm.update_record_stimulation_dynamic_model_right(
            ampli=[[75000, -75000, 0]],
            duri=[[200, 200, 9600]],
        ),
        1.5,
        "right",
    ),
}


def _check_active(stim_pool, expected_side, left_ele, right_ele):
    """断言激活集合与期望一致（防 DAC0 串扰）。"""
    active = stim_pool.get_active_electrodes()
    if expected_side == "left":
        expected = {left_ele}
    elif expected_side == "right":
        expected = {right_ele}
    elif expected_side == "both":
        expected = {left_ele, right_ele}
    else:
        return  # 未知 side 不校验
    if active != expected:
        print("  ! WARN: active electrodes {} != expected {} (DAC0 crosstalk?)".format(
            sorted(active), sorted(expected)
        ))
    else:
        print("  active electrodes = {} (matches expected '{}')".format(
            sorted(active), expected_side
        ))


def main():
    parser = argparse.ArgumentParser(
        description="Maxwell left/right wheel stim business-layer smoke."
    )
    parser.add_argument("cfg_path", help="MaxLab Live exported .cfg path")
    parser.add_argument("--left", type=int, required=True,
                        help="Left wheel stim electrode ID")
    parser.add_argument("--right", type=int, required=True,
                        help="Right wheel stim electrode ID")
    parser.add_argument("--tests", default="all",
                        help="Comma-separated test names or 'all'. "
                             "Available: " + ",".join(TEST_REGISTRY.keys()))
    parser.add_argument("--gap", type=float, default=2.0,
                        help="Inter-test gap seconds (default 2.0)")
    args = parser.parse_args()

    cfg_path = Path(args.cfg_path)
    if not cfg_path.exists():
        print("cfg not found: {}".format(cfg_path))
        sys.exit(2)

    if args.left == args.right:
        print("--left and --right must differ")
        sys.exit(2)

    if args.tests.lower() == "all":
        test_names = list(TEST_REGISTRY.keys())
    else:
        test_names = [t.strip() for t in args.tests.split(",") if t.strip()]
        unknown = [t for t in test_names if t not in TEST_REGISTRY]
        if unknown:
            print("unknown tests: {}. available: {}".format(
                unknown, list(TEST_REGISTRY.keys())
            ))
            sys.exit(2)

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    print("cfg_path                    = {}".format(cfg_path))
    print("left wheel electrode        = {}".format(args.left))
    print("right wheel electrode       = {}".format(args.right))
    print("tests                       = {}".format(test_names))
    print("inter-test gap              = {:.1f}s".format(args.gap))

    from src.system_device.maxwell_sys import MaxwellSystem

    sys_obj = MaxwellSystem()
    failed = False

    _print_banner("[1/3] start_session")
    try:
        sys_obj.start_session(
            cfg_path=str(cfg_path),
            record_electrodes=None,
            stim_electrodes=[args.left, args.right],
            role_mapping=None,
        )
        stm = sys_obj.stimulating
        print("  left_wheel : electrode={} unit={}".format(
            stm.left_electrode, stm.left_unit_id
        ))
        print("  right_wheel: electrode={} unit={}".format(
            stm.right_electrode, stm.right_unit_id
        ))
        print("  DAC LSB    = {:.4f} mV/bit".format(stm._dac_lsb_mv))
    except Exception:
        failed = True
        traceback.print_exc()
        try:
            sys_obj.stop_connect()
        except Exception:
            pass
        print()
        print("RESULT: FAIL (start_session)")
        sys.exit(1)

    _print_banner("[2/3] run stim tests")
    try:
        for i, name in enumerate(test_names, start=1):
            desc, fn, sleep_s, expected_side = TEST_REGISTRY[name]
            print()
            print("--- [{}/{}] {} ---".format(i, len(test_names), name))
            print("  {}".format(desc))
            t0 = time.time()
            fn(stm)
            elapsed = time.time() - t0
            print("  fired in {:.3f}s".format(elapsed))
            _check_active(sys_obj.stim_pool, expected_side, args.left, args.right)
            _wait(sleep_s, "let sequence complete")
            if i < len(test_names):
                _wait(args.gap, "inter-test gap")
    except Exception:
        failed = True
        traceback.print_exc()

    _print_banner("[3/3] stop_connect")
    try:
        sys_obj.stop_connect()
        print("  stop_connect OK")
    except Exception as exc:
        print("  stop_connect raised: {!r}".format(exc))

    print()
    if failed:
        print("RESULT: FAIL")
        sys.exit(1)
    print("RESULT: PASS")
    print()
    print("如果 MaxLab Live 同时在录制，frame 元数据应能看到 Event labels:")
    print("  env_left / env_right / punish_left / punish_right /")
    print("  reward_left / reward_right / reward_both / mpc_left / mpc_right")
    print("各对应电极上可见 200us 双相波形（reward 为 -75/+75 mV，env 为 +75/-75 mV）。")


if __name__ == "__main__":
    main()
