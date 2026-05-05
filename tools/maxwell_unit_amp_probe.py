"""
Maxwell stim_unit -> amplifier 反向枚举脚本（找 stim 电极的真 unit_id）。

背景
----
lr_smoke 跑 `--tests env_right,env_right,env_right` 后实测：amp 404 (left) 全程
变化、amp 147 (right) 全程不变。叠加证据：
  - 之前 connect_probe P0 baseline:
      query_stimulation_at_electrode(18884) = '26'
      query_amplifier_at_stimulation(unit=2) = ''     ← unit 2 不连
      query_amplifier_at_electrode(18884) = '147'    ← right 实际在 amp 147
  - 修复后 lr_smoke recording_maxwell.connect:
      mapped electrode=18884 -> unit=2 (post-download)  ← API 误导

`query_stimulation_at_electrode` 在不同跑次返回值不一致，不能作为 ground truth。
真正的 ground truth API 是 `query_amplifier_at_stimulation(unit_id)`：返回 unit 实际
连接的 amplifier channel；与 `query_amplifier_at_electrode(electrode)` 的返回值
匹配，即可定位该 stim 电极对应的真 unit_id。

本脚本完成
----------
1. 用 MaxwellSystem.start_session 走完整启动序列（含 download + offset）
2. download 后枚举 unit 0-31，调 query_amplifier_at_stimulation 打印每个 unit
   实际连的 amp_channel
3. 同步打印 stim 电极 query_amplifier_at_electrode 与 query_stimulation_at_electrode
   两个返回值供对照
4. 最后表格化输出「electrode → real unit (反查) vs API 缓存 unit (api 直查)」差异

跑完此脚本即可定下 right 电极的真 unit_id，作为后续 stim_pool 修复的依据。

使用
----
    /home/maxwell/metaboc-env/bin/python tools/maxwell_unit_amp_probe.py \\
        /home/maxwell/configs/260227/pi_16h44m18s.cfg \\
        --left 2661 --right 18884
"""

import argparse
import sys
from pathlib import Path


MAX_STIM_UNITS = 32


def _print_banner(title):
    print()
    print("=" * 72)
    print("  " + title)
    print("=" * 72)


def _safe_str(rv):
    if rv is None:
        return "<None>"
    return str(rv)


def main():
    parser = argparse.ArgumentParser(
        description="Maxwell stim_unit -> amp reverse enumeration probe."
    )
    parser.add_argument("cfg_path", help="MaxLab Live exported .cfg path")
    parser.add_argument("--left", type=int, required=True,
                        help="Left wheel stim electrode ID")
    parser.add_argument("--right", type=int, required=True,
                        help="Right wheel stim electrode ID")
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

    from src.system_device.maxwell_sys import MaxwellSystem

    sys_obj = MaxwellSystem()

    _print_banner("[1/4] start_session (含 download + offset)")
    sys_obj.start_session(
        cfg_path=str(cfg_path),
        record_electrodes=None,
        stim_electrodes=[args.left, args.right],
        role_mapping=None,
    )
    array = sys_obj.stim_pool._array
    cached = dict(sys_obj.recording._stim_electrode_to_unit)
    print()
    print("recording_maxwell cached mapping (api 直查): {}".format(cached))

    _print_banner("[2/4] 每个 stim 电极的 amp 与 stim_unit (api 直查)")
    electrode_to_amp = {}
    print("  {:<10s} {:<10s} {:<10s} {:<25s}".format(
        "electrode", "amp_chan", "api_unit", "(api_unit 来自 query_stimulation_at_electrode)"
    ))
    for ele in [args.left, args.right]:
        amp = array.query_amplifier_at_electrode(ele)
        api_unit = array.query_stimulation_at_electrode(ele)
        electrode_to_amp[ele] = _safe_str(amp).strip()
        print("  {:<10d} {:<10s} {:<10s}".format(
            ele, _safe_str(amp), _safe_str(api_unit)
        ))

    _print_banner("[3/4] 枚举 unit 0-31: query_amplifier_at_stimulation (ground truth 反查)")
    unit_to_amp = {}
    print("  {:<8s} {:<10s}".format("unit_id", "amp_chan"))
    for unit_id in range(MAX_STIM_UNITS):
        try:
            unit_amp = array.query_amplifier_at_stimulation(unit_id)
        except Exception as e:
            print("  unit {} raised {!r}".format(unit_id, e))
            continue
        unit_amp_s = _safe_str(unit_amp).strip()
        unit_to_amp[unit_id] = unit_amp_s
        if unit_amp_s == "" or unit_amp_s == "<None>":
            tag = "(empty)"
        else:
            tag = ""
        print("  {:<8d} {:<10s} {}".format(unit_id, unit_amp_s, tag))

    _print_banner("[4/4] 比对：electrode 真 unit (反查) vs api_unit (直查)")
    print("  {:<10s} {:<10s} {:<14s} {:<14s} {:<10s}".format(
        "electrode", "amp_chan", "real_unit", "api_unit", "match?"
    ))
    for ele in [args.left, args.right]:
        ele_amp = electrode_to_amp[ele]
        real_unit_candidates = [
            u for u, a in unit_to_amp.items() if a == ele_amp and ele_amp not in ("", "<None>")
        ]
        real_unit = real_unit_candidates[0] if real_unit_candidates else None
        api_unit_str = _safe_str(array.query_stimulation_at_electrode(ele)).strip()
        if real_unit is None:
            match = "NONE"
        else:
            try:
                api_unit_int = int(api_unit_str)
                match = "OK" if api_unit_int == real_unit else "MISMATCH"
            except ValueError:
                match = "MISMATCH(api 非数字)"
        print("  {:<10d} {:<10s} {:<14s} {:<14s} {:<10s}".format(
            ele, ele_amp, str(real_unit), api_unit_str, match
        ))

    _print_banner("cleanup — stop_connect")
    try:
        sys_obj.stop_connect()
        print("  stop_connect OK")
    except Exception as exc:
        print("  stop_connect raised: {!r}".format(exc))

    print()
    print("=" * 72)
    print("  结论用法")
    print("=" * 72)
    print("  - 看 [4/4] 表格的 real_unit 列：right (18884) 应该有一个非 None 的 unit_id")
    print("  - 该 unit_id 即为 stim_pool 应该使用的 right 真 unit")
    print("  - 若 api_unit 与 real_unit 不一致 (MISMATCH)，则 query_stimulation_at_electrode")
    print("    确认不可信，stim_pool 缓存策略需改为「query_amplifier_at_stimulation 反查」")
    print("  - 若两次跑得到的 real_unit 相同，则 query_amplifier_at_stimulation 稳定可作 ground truth")


if __name__ == "__main__":
    main()
