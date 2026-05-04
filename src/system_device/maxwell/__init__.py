"""
Maxwell (MaxOne / MaxTwo) device integration sub-package.

设备无关基础设施层。本包内所有模块禁止在顶层 import maxlab，
maxlab 必须在函数内部 lazy import，保证 Windows 开发机不依赖
Maxwell SDK 即可通过 py_compile 与 import smoke test。

约束依据：[[Maxwell - 平台接入开发约束]]
"""
