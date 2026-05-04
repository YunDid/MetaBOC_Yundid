"""
全局平台检测与 MCS 可用性探测。

MetaBOC 平台原生仅支持 Windows，因 MCS MEA2100 硬件线依赖 Windows 专属
的 McsUsbNet.dll（通过 pythonnet CLR 互操作加载 .NET DLL）。本模块在
导入时一次性确定平台与 MCS 可用性，供其他模块在 import 顶层做条件守卫。

约定：
- IS_WINDOWS=True  且 MCS_AVAILABLE=True  → MCS 路径完全启用
- IS_WINDOWS=True  且 MCS_AVAILABLE=False → Windows 但 pythonnet 未装，跳过 MCS
- IS_WINDOWS=False                         → Linux/Mac，跳过 MCS

依据：[[MetaBOC - 从 Windows 到 Linux 的 MCS 依赖隔离与条件分支方案]]
"""

import sys

IS_WINDOWS = sys.platform == "win32"

MCS_AVAILABLE = False
if IS_WINDOWS:
    try:
        import clr  # noqa: F401  仅探测可用性
        MCS_AVAILABLE = True
    except ImportError:
        MCS_AVAILABLE = False
