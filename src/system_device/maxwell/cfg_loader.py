"""
cfg 文件解析与电极存在性校验。

MetaBOC 平台不实现活性扫描 / 网络检测。这些工作由 MaxLab Live 上位机
软件完成，导出 .cfg 文件作为平台输入。本模块负责把 cfg 文本解析成
电极列表，交给上层用 array.select_electrodes / select_stimulation_electrodes
显式 routing，平台不直接调用 Array.load_config（避免黑盒加载，路由权
完全保留在平台层）。

cfg 文件格式（MaxLab Live 导出的纯文本）：
    每条记录形如  "<channel>(<electrode>)<x>/<y>"，多条以空白分隔。

接口约定：
- parse_cfg(cfg_path)         返回 [{"channel", "electrode", "x", "y"}, ...]
- extract_electrodes(cfg_path) 返回去重保序的 electrode ID 列表
- validate_record_electrodes(array, expected) 校验 routing 结果覆盖
"""

import os
import re
from pathlib import Path
from typing import Dict, List

from .errors import ConfigError


_CFG_ENTRY_PATTERN = re.compile(r"(\d+)\((\d+)\)([\d.]+)/([\d.]+)")


def _read_cfg_text(cfg_path):
    """读 cfg 文件并做基础校验。返回纯文本内容。"""
    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise ConfigError("cfg file does not exist: {}".format(cfg_path))
    if not cfg_path.is_file():
        raise ConfigError("cfg path is not a regular file: {}".format(cfg_path))
    if not os.access(str(cfg_path), os.R_OK):
        raise ConfigError("cfg file is not readable: {}".format(cfg_path))

    with open(str(cfg_path), "r", encoding="utf-8") as handle:
        return handle.read().strip(), cfg_path.resolve()


def parse_cfg(cfg_path) -> List[Dict]:
    """
    解析 MaxLab Live 导出的 cfg 文件，返回每条 channel/electrode/坐标记录。

    Returns
    -------
    list[dict]
        [{"channel": int, "electrode": int, "x": float, "y": float}, ...]

    Raises
    ------
    ConfigError
        当 cfg 路径不可读、内容无法匹配 cfg 格式、或 entries 为空时。
    """
    raw, cfg_abs = _read_cfg_text(cfg_path)

    entries: List[Dict] = []
    for match in _CFG_ENTRY_PATTERN.finditer(raw):
        entries.append({
            "channel": int(match.group(1)),
            "electrode": int(match.group(2)),
            "x": float(match.group(3)),
            "y": float(match.group(4)),
        })

    if not entries:
        raise ConfigError(
            "cfg parse produced 0 entries. File may be empty or format mismatch: {}".format(cfg_abs)
        )

    return entries


def extract_electrodes(cfg_path) -> List[int]:
    """
    从 cfg 解析出唯一 electrode ID 列表，按首次出现顺序保留。

    返回的列表直接喂给 `Array.select_electrodes(electrodes)`。stim 电极
    通常是该列表的子集，由上层显式声明，不从 cfg 推断。
    """
    seen = set()
    electrodes: List[int] = []
    for entry in parse_cfg(cfg_path):
        electrode = entry["electrode"]
        if electrode in seen:
            continue
        seen.add(electrode)
        electrodes.append(electrode)
    return electrodes


def validate_record_electrodes(array, expected_electrodes):
    """
    校验 routing 完成后期望的记录电极是否都已 routed 到放大器。

    cfg 文件可能与当前实验设计不一致，select_electrodes + route 后
    某些电极可能因冲突未能 routed。在闭环开始前必须用
    query_amplifier_at_electrode 严格校验，否则 recording 会读到空通道。

    Parameters
    ----------
    array : maxlab.chip.Array
        已经 route + download 完成的 Array 对象。
    expected_electrodes : list[int]
        期望参与记录的电极物理 ID 列表。

    Returns
    -------
    dict
        {
            "routed":     [electrode_ids 已路由],
            "missing":    [electrode_ids 未路由],
            "amplifiers": {electrode_id: amplifier_channel},
        }

    Raises
    ------
    ConfigError
        当存在任何 missing 电极时（严格校验，禁止部分覆盖运行）。
    """
    routed = []
    missing = []
    amplifiers = {}

    for electrode in expected_electrodes:
        try:
            amp = array.query_amplifier_at_electrode(electrode)
        except Exception as exc:
            raise ConfigError(
                "query_amplifier_at_electrode failed for electrode={}: {!r}".format(electrode, exc)
            )

        if amp is None or (hasattr(amp, "__len__") and len(amp) == 0):
            missing.append(electrode)
        else:
            routed.append(electrode)
            # query_amplifier_at_electrode 实测返回字符串（如 '404' / '147'），
            # 不是 list/tuple。原 hasattr(__len__) ? int(amp[0]) 分支对字符串会切到
            # 首字符（'404'[0]='4' → int=4），在 amp_chan ≥ 10 时悄无声息错位。
            # 与 recording_maxwell.connect 中 query_stimulation_at_electrode 的同模式
            # BUG 一并修复（commit db4501b 已修 stim_unit 一侧）。
            if isinstance(amp, (list, tuple)):
                amplifiers[electrode] = int(amp[0])
            else:
                amplifiers[electrode] = int(amp)

    report = {"routed": routed, "missing": missing, "amplifiers": amplifiers}

    if missing:
        raise ConfigError(
            "cfg does not cover {} expected record electrodes: {}".format(len(missing), missing)
        )

    return report
