#!/usr/bin/env python3
"""
Maxwell 接入 stub 分层验证脚本。

用法
----
基础验证（无需任何 Maxwell 依赖）：
    python scripts/maxwell_verify.py

带 cfg 文件的完整启动验证：
    python scripts/maxwell_verify.py --cfg /path/to/your.cfg \\
        --record 11110,11111,11112 --stim 15000

每一层独立，前一层失败时后续层自动跳过。退出码：
    0 = 至少 Tier 1 通过；命令行未要求的层跳过不算失败
    1 = Tier 1 失败（关键依赖缺失或导入错误）
"""

import argparse
import os
import sys
import traceback


# 把项目根目录加入 sys.path，以便从 scripts/ 子目录运行能 import src.*
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def banner(text):
    bar = "=" * 70
    print("\n" + bar)
    print("  " + text)
    print(bar)


def tier1_import_check():
    """Tier 1：仅 Python 标准库 + 项目内模块 import。无任何外部依赖。"""
    banner("Tier 1 / 模块 import 与跨平台兼容性")

    try:
        from src.robot.task import SYSTEM_DEVICE
        print("  [OK] src.robot.task.SYSTEM_DEVICE imported")
        print("       values: {}".format([x.name for x in SYSTEM_DEVICE]))
        assert "MAXWELL" in [x.name for x in SYSTEM_DEVICE], "MAXWELL enum missing"
        print("  [OK] MAXWELL enum value present")
    except Exception as exc:
        print("  [FAIL] task.py import: {!r}".format(exc))
        return False

    try:
        from src.system_device.maxwell import (
            errors, session_lifecycle, cfg_loader, stim_pool,
        )
        print("  [OK] maxwell sub-package modules imported (errors / session_lifecycle / cfg_loader / stim_pool)")
    except Exception as exc:
        print("  [FAIL] maxwell sub-package import: {!r}".format(exc))
        return False

    try:
        from src.system_device.maxwell.recording_maxwell import RecordingMaxwell
        from src.system_device.maxwell.stimulation_maxwell import StimulationMaxwell
        from src.system_device.maxwell_sys import MaxwellSystem
        print("  [OK] RecordingMaxwell / StimulationMaxwell / MaxwellSystem imported")
    except Exception as exc:
        print("  [FAIL] role + system import: {!r}".format(exc))
        return False

    try:
        sys_obj = MaxwellSystem()
        assert type(sys_obj.recording).__name__ == "RecordingMaxwell"
        assert type(sys_obj.stimulating).__name__ == "StimulationMaxwell"
        print("  [OK] MaxwellSystem instantiable; recording/stimulating wired")
    except Exception as exc:
        print("  [FAIL] MaxwellSystem instantiation: {!r}".format(exc))
        return False

    return True


def tier2_maxlab_check():
    """Tier 2：maxlab Python 包是否已安装。"""
    banner("Tier 2 / maxlab Python 包")

    try:
        import maxlab as mx
        print("  [OK] maxlab imported")
        print("       maxlab module path: {}".format(getattr(mx, "__file__", "<built-in>")))

        for attr in ("initialize", "send", "Core", "Timing", "activate",
                     "Array", "offset", "clear_events", "StimulationUnit"):
            present = hasattr(mx, attr) or hasattr(getattr(mx, "chip", None), attr)
            mark = "[OK]" if present else "[WARN]"
            print("  {} maxlab.{} present".format(mark, attr))

        for timing_const in ("waitInit", "waitAfterDownload",
                             "waitInMX2Offset", "waitAfterRecording"):
            present = hasattr(getattr(mx, "Timing", object()), timing_const)
            mark = "[OK]" if present else "[WARN]"
            print("  {} maxlab.Timing.{} present".format(mark, timing_const))

        return True
    except ImportError as exc:
        print("  [SKIP] maxlab not installed: {}".format(exc))
        print("         Install via Maxwell SDK pip wheel or source.")
        return None
    except Exception as exc:
        print("  [FAIL] unexpected error importing maxlab: {!r}".format(exc))
        return False


def tier3_mxwserver_probe():
    """Tier 3：mxwserver 进程是否可达。"""
    banner("Tier 3 / mxwserver 探活")

    try:
        from src.system_device.maxwell.session_lifecycle import check_mxwserver_alive
        from src.system_device.maxwell.errors import MxwserverError

        try:
            check_mxwserver_alive()
            print("  [OK] mxwserver is reachable and responded 'Ok'")
            return True
        except MxwserverError as exc:
            print("  [FAIL] mxwserver probe failed: {}".format(exc))
            print("         Make sure mxwserver is running on this machine.")
            return False
    except ImportError as exc:
        print("  [SKIP] cannot probe (maxlab missing): {}".format(exc))
        return None


def tier4_session_lifecycle(cfg_path, record_electrodes, stim_electrodes):
    """Tier 4：完整启动序列 + cfg 加载 + stim pool 上电 + 清理。"""
    banner("Tier 4 / 完整启动序列（cfg + record + stim + offset + cleanup）")

    if cfg_path is None:
        print("  [SKIP] no --cfg argument supplied; cannot validate session lifecycle")
        return None

    if not os.path.exists(cfg_path):
        print("  [FAIL] cfg file does not exist: {}".format(cfg_path))
        return False

    print("  cfg path:           {}".format(cfg_path))
    print("  record electrodes:  {}".format(record_electrodes))
    print("  stim electrodes:    {}".format(stim_electrodes))

    try:
        from src.system_device.maxwell_sys import MaxwellSystem
    except Exception as exc:
        print("  [FAIL] cannot import MaxwellSystem: {!r}".format(exc))
        return False

    sys_obj = MaxwellSystem()

    role_mapping = {}
    if stim_electrodes:
        for i, electrode in enumerate(stim_electrodes):
            role_mapping["stim_{}".format(i)] = electrode

    try:
        sys_obj.start_session(
            cfg_path=cfg_path,
            record_electrodes=record_electrodes,
            stim_electrodes=stim_electrodes,
            role_mapping=role_mapping,
        )
        print("  [OK] start_session completed without exception")
    except Exception as exc:
        print("  [FAIL] start_session raised: {!r}".format(exc))
        traceback.print_exc()
        try:
            sys_obj.stop_connect()
        except Exception:
            pass
        return False

    try:
        spikes = sys_obj.recording.get_recording()
        print("  [OK] get_recording returned: {} (Phase B stub returns empty lists)".format(spikes))
    except Exception as exc:
        print("  [FAIL] get_recording raised: {!r}".format(exc))

    try:
        sys_obj.stimulating.update_record_stimulation(0.5, 0.5)
        print("  [OK] update_record_stimulation invoked (stub prints message)")
    except Exception as exc:
        print("  [FAIL] update_record_stimulation raised: {!r}".format(exc))

    try:
        sys_obj.stop_connect()
        print("  [OK] stop_connect completed")
    except Exception as exc:
        print("  [FAIL] stop_connect raised: {!r}".format(exc))
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cfg", default=None,
                        help="Maxwell cfg 文件路径，触发 Tier 4 完整启动验证")
    parser.add_argument("--record", default=None,
                        help="逗号分隔的记录电极 ID（如 '11110,11111,11112'）")
    parser.add_argument("--stim", default=None,
                        help="逗号分隔的刺激电极 ID（如 '15000,15500'）")
    args = parser.parse_args()

    record_electrodes = None
    if args.record:
        record_electrodes = [int(x.strip()) for x in args.record.split(",") if x.strip()]
    stim_electrodes = None
    if args.stim:
        stim_electrodes = [int(x.strip()) for x in args.stim.split(",") if x.strip()]

    results = {}

    results["tier1"] = tier1_import_check()
    if not results["tier1"]:
        print("\n[ABORT] Tier 1 failed. Subsequent tiers depend on it.")
        sys.exit(1)

    results["tier2"] = tier2_maxlab_check()
    if results["tier2"] is False:
        print("\n[ABORT] Tier 2 had unexpected failure.")
        sys.exit(1)

    if results["tier2"] is None:
        results["tier3"] = None
        results["tier4"] = None
    else:
        results["tier3"] = tier3_mxwserver_probe()
        if results["tier3"] is True:
            results["tier4"] = tier4_session_lifecycle(args.cfg, record_electrodes, stim_electrodes)
        else:
            results["tier4"] = None

    banner("Verification Summary")
    label = {True: "PASS", False: "FAIL", None: "SKIP"}
    for tier in ("tier1", "tier2", "tier3", "tier4"):
        print("  {}: {}".format(tier.upper(), label[results[tier]]))

    sys.exit(0)


if __name__ == "__main__":
    main()
