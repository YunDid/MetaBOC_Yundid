"""
Maxwell Phase B 接入烟雾测试 (smoke test)。

目的
----
不依赖 GUI、不依赖完整 main.py，最小路径验证 Maxwell 后端骨架是否闭合：
  - cfg 解析 → 显式 select_electrodes / select_stimulation_electrodes
  - 8 步初始化 + route + download + offset 校正
  - stim_pool 查 unit + 上电
  - left_wheel / right_wheel 角色绑定（stim_electrodes[0] / [1]）
  - get_recording 占位返回
  - stimulation 各 update_* 接口签名暴露完整
  - close 容错清理

**不发真实刺激**。真实刺激发放由 tools/maxwell_stim_smoke.py 单独验证。

使用
----
Linux 真机（mxwserver 运行中）：

    python tools/maxwell_phase_b_smoke.py /path/to/your.cfg

  - 默认 stim_electrodes = []，仅验证 recording 路径
  - 加 --stim "3580,4887" 验证 stim 完整链路 + left/right wheel 绑定
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

        # 仅验证 stimulation 接口暴露完整（不真发刺激；真发由 maxwell_stim_smoke.py 验证）
        expected_methods = [
            "update_record_stimulation",
            "update_stimulation_left",
            "update_stimulation_right",
            "update_stimulation_left_reward",
            "update_stimulation_right_reward",
            "update_stimulation_left_right_reward",
            "update_record_stimulation_dynamic_model_left",
            "update_record_stimulation_dynamic_model_right",
        ]
        missing = [m for m in expected_methods
                   if not callable(getattr(sys_obj.stimulating, m, None))]
        if missing:
            raise RuntimeError("stimulation missing methods: {}".format(missing))
        print("[A4] stimulation API surface OK ({} update_* methods exposed; no live stim).".format(
            len(expected_methods)
        ))

        if stim_electrodes and len(stim_electrodes) >= 2:
            stm = sys_obj.stimulating
            assert stm.left_electrode == stim_electrodes[0], \
                "left_wheel binding mismatch: expected {}, got {}".format(
                    stim_electrodes[0], stm.left_electrode)
            assert stm.right_electrode == stim_electrodes[1], \
                "right_wheel binding mismatch: expected {}, got {}".format(
                    stim_electrodes[1], stm.right_electrode)
            print("[A4b] left/right wheel binding verified: "
                  "left={} (unit {}), right={} (unit {})".format(
                      stm.left_electrode, stm.left_unit_id,
                      stm.right_electrode, stm.right_unit_id))

    finally:
        try:
            sys_obj.stop_connect()
            print("[A5] stop_connect OK.")
        except Exception as exc:
            print("[A5] stop_connect raised: {!r}".format(exc))


def smoke_tier_b(cfg_path, stim_electrodes, role_mapping):
    """通过 Communication 包装层，模拟 GUI 路径。"""
    _print_banner("Tier B — Communication wrapper")

    # Communication.__init__ 实例化 EncodingDecoding(QDialog)，
    # 必须先有 QApplication 在场，否则 Qt 直接 abort。
    # 与真实 GUI 启动路径一致——main.py 同样先 QApplication 再 Communication。
    from PyQt5.QtWidgets import QApplication
    _qt_app = QApplication.instance() or QApplication([])

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

        # 仅验证 Communication 与 Maxwell stimulation 角色对象绑定成功
        # （不真发刺激；真发由 maxwell_stim_smoke.py 验证）
        if not callable(getattr(comm.stimulation, "update_record_stimulation", None)):
            raise RuntimeError("Communication.stimulation missing update_record_stimulation")
        print("[B6] stimulation interface bound on Communication (no live stim send).")

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
