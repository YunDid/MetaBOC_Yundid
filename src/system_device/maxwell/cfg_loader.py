"""
cfg 文件加载与电极存在性校验。

MetaBOC 平台不实现活性扫描 / 网络检测。这些工作由 MaxLab Live 上位机
软件完成，导出 .cfg 文件作为平台输入。本模块负责把 cfg 安全加载到
mxwserver 中，并校验后续实验需要的电极是否已经被路由到放大器。
"""

import os
from pathlib import Path

from .errors import ConfigError


def load_config(array, cfg_path):
    """
    从磁盘加载 cfg 文件到指定 Array 对象。

    Parameters
    ----------
    array : maxlab.chip.Array
        已经创建的 Array 对象。函数内部会先 reset 再 load。
    cfg_path : str or Path
        cfg 文件路径。必须存在且可读。

    Returns
    -------
    Path
        实际加载的 cfg 文件绝对路径，用于上层留痕。

    Raises
    ------
    ConfigError
        当 cfg 路径不存在、无权访问、或加载失败时。
    """
    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise ConfigError("cfg file does not exist: {}".format(cfg_path))
    if not cfg_path.is_file():
        raise ConfigError("cfg path is not a regular file: {}".format(cfg_path))
    if not os.access(str(cfg_path), os.R_OK):
        raise ConfigError("cfg file is not readable: {}".format(cfg_path))

    cfg_abs = cfg_path.resolve()

    try:
        array.reset()
        array.load_config(str(cfg_abs))
    except Exception as exc:
        raise ConfigError(
            "Failed to load cfg into array: path={} error={!r}".format(cfg_abs, exc)
        )

    return cfg_abs


def validate_record_electrodes(array, expected_electrodes):
    """
    校验 cfg 加载后期望的记录电极是否已经被路由到放大器。

    cfg 文件可能来自历史实验、对应的电极池可能与当前实验设计不一致。
    在闭环开始前必须用 query_amplifier_at_electrode 确认每个目标
    电极都已 routed，否则 recording 会读到空通道。

    Parameters
    ----------
    array : maxlab.chip.Array
        已经 load_config 完成的 Array 对象。
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
            amplifiers[electrode] = int(amp) if not hasattr(amp, "__len__") else int(amp[0])

    report = {"routed": routed, "missing": missing, "amplifiers": amplifiers}

    if missing:
        raise ConfigError(
            "cfg does not cover {} expected record electrodes: {}".format(len(missing), missing)
        )

    return report
