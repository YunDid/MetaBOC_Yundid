"""
Maxwell Phase B 接入烟雾测试 (smoke test)。

目的
----
不依赖 GUI、不依赖完整 main.py，最小路径验证 Maxwell 后端骨架是否闭合：
  - cfg 解析 → 显式 select_electrodes / select_stimulation_electrodes
  - 8 步初始化 + route + download + offset 校正
  - stim_pool 查 unit + 上电
  - get_recording 占位返回
  - stimulation 各 update_* 接口 print 路径
  - close 容错清理

每一步打印状态，挂在哪一步一眼可见。

使用
----
Linux 真机（mxwserver 运行中）：

    python tools/maxwell_phase_b_smoke.py /path/to/your.cfg

  - 默认 stim_electrodes = []，仅验证 recording 路径
  - 加 --stim "3580,4887" 验证 stim 完整链路
  - 加 --tier B 经过 Communication 包装层
"""

import argparse
import sys
import traceback
from pathlib import Path


def _print_banner(title):
    print()
    print("=" * 60)
    print("  " + title)
    print("=" * 60)


def _parse_int_list(value):
    if not value:
        return []
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def smoke_tier_a(cfg_path, stim_electrodes, role_mapping):
    """直连 MaxwellSystem，绕过 Communication。"""
    _print_banner("Tier A — MaxwellSystem direct")

    from src.system_device.maxwell_sys import MaxwellSystem

    sys_obj = MaxwellSystem()
    print("[A1] MaxwellSystem instantiated.")

    try:
        sys_obj.start_session(
            cfg_path=str(cfg_path),
            record_electrodes=None,    # 走 cfg 解析路径
            stim_electrodes=stim_electrodes,
            role_mapping=role_mapping,
        )
        print("[A2] start_session OK. recording_connected={}, stim_pool={}".format(
            sys_obj.recording._connected,
            "yes" if sys_obj.stim_pool is not None else "no",
        ))

        left, right = sys_obj.recording.get_recording()
        print("[A3] get_recording returned: left={} right={} (Phase B stub expects [], [])".format(
            left, right
        ))

        sys_obj.stimulating.update_record_stimulation(left=10, right=20)
        sys_obj.stimulating.update_stimulation_left()
        sys_obj.stimulating.update_stimulation_right()
        sys_obj.stimulating.update_stimulation_left_right_reward()
        sys_obj.stimulating.update_record_stimulation_dynamic_model_left(ampli=[100], duri=[200])
        sys_obj.stimulating.update_record_stimulation_dynamic_model_right(ampli=[100], duri=[200])
        print("[A4] stimulation API surface OK (printed messages).")

    finally:
        try:
            sys_obj.stop_connect()
            print("[A5] stop_connect OK.")
        except Exception as exc:
            print("[A5] stop_connect raised: {!r}".format(exc))


def smoke_tier_b(cfg_path, stim_electrodes, role_mapping):
    """通过 Communication 包装层，模拟 GUI 路径。"""
    _print_banner("Tier B — Communication wrapper")

    from src.robot.communication import Communication
    from src.robot.task import SYSTEM_DEVICE

    comm = Communication()
    print("[B1] Communication instantiated. device_type={}".format(comm.device_type))

    try:
        comm.set_maxwell_session_params(
            cfg_path=str(cfg_path),
            record_electrodes=None,
            stim_electrodes=stim_electrodes,
            role_mapping=role_mapping,
        )
        print("[B2] set_maxwell_session_params OK.")

        comm.update_systems(SYSTEM_DEVICE.MAXWELL)
        print("[B3] update_systems(MAXWELL) OK. device_type={}".format(comm.device_type))

        if comm.recording is None or comm.stimulation is None:
            print("[B4] FAIL: recording/stimulation references not bound.")
            return
        print("[B4] role objects bound: recording={} stimulation={}".format(
            type(comm.recording).__name__,
            type(comm.stimulation).__name__,
        ))

        left, right = comm.recording.get_recording()
        print("[B5] get_recording returned: left={} right={}".format(left, right))

        comm.stimulation.update_record_stimulation(left=10, right=20)
        print("[B6] stimulation update_record_stimulation OK.")

    finally:
        try:
            comm.close()
            print("[B7] Communication.close OK.")
        except Exception as exc:
            print("[B7] close raised: {!r}".format(exc))


def main():
    parser = argparse.ArgumentParser(description="Maxwell Phase B smoke test.")
    parser.add_argument("cfg_path", help="Path to MaxLab Live exported .cfg file")
    parser.add_argument("--stim", default="",
                        help="Comma-separated stim electrode IDs (e.g. '3580,4887'). "
                             "Empty = recording-only path.")
    parser.add_argument("--tier", default="A", choices=["A", "B", "AB"],
                        help="A = MaxwellSystem direct (default); "
                             "B = via Communication; AB = both.")
    parser.add_argument("--role-left", type=int, default=None,
                        help="env_left electrode id (must be in --stim).")
    parser.add_argument("--role-right", type=int, default=None,
                        help="env_right electrode id (must be in --stim).")
    args = parser.parse_args()

    cfg_path = Path(args.cfg_path)
    if not cfg_path.exists():
        print("cfg file does not exist: {}".format(cfg_path))
        sys.exit(2)

    stim_electrodes = _parse_int_list(args.stim)
    role_mapping = {}
    if args.role_left is not None:
        role_mapping["env_left"] = args.role_left
    if args.role_right is not None:
        role_mapping["env_right"] = args.role_right

    print("cfg_path        = {}".format(cfg_path))
    print("stim_electrodes = {}".format(stim_electrodes))
    print("role_mapping    = {}".format(role_mapping))
    print("tier            = {}".format(args.tier))

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    failed = False
    try:
        if "A" in args.tier:
            smoke_tier_a(cfg_path, stim_electrodes, role_mapping or None)
        if "B" in args.tier:
            smoke_tier_b(cfg_path, stim_electrodes, role_mapping or None)
    except Exception:
        failed = True
        print()
        print("SMOKE FAILED — traceback below:")
        traceback.print_exc()

    print()
    if failed:
        print("RESULT: FAIL")
        sys.exit(1)
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
