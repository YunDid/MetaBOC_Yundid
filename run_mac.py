"""
Mac 浅色启动器（新增文件，不修改 main.py / 任何现有代码）。

问题：本 App 的 .ui 与 main.py 只写死了浅色背景，从不显式设文字色；
macOS 暗色模式下 Qt 默认调色板把 WindowText 取成白色 → 白字落在写死的
浅背景上不可读。系统外观开关时灵时不灵，故在此用启动器层面与系统主题
彻底解耦：建窗口前强制 Fusion 风格 + 一套完整浅色调色板。

用法：
    cd /Users/yundid/workspace/MetaBOC_Yundid
    /Users/yundid/miniconda3/envs/metaboc/bin/python run_mac.py
"""

import os
import sys
import runpy

# main.py 用相对路径（如 ./out/MEA2100），把工作目录钉死到本文件所在仓库根
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPalette, QColor

app = QApplication(sys.argv)
app.setStyle("Fusion")

_DARK = QColor(20, 20, 20)
pal = QPalette()
pal.setColor(QPalette.Window, QColor(248, 248, 248))
pal.setColor(QPalette.WindowText, _DARK)
pal.setColor(QPalette.Base, QColor(255, 255, 255))
pal.setColor(QPalette.AlternateBase, QColor(240, 240, 240))
pal.setColor(QPalette.Text, _DARK)
pal.setColor(QPalette.Button, QColor(240, 240, 240))
pal.setColor(QPalette.ButtonText, _DARK)
pal.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
pal.setColor(QPalette.ToolTipText, _DARK)
pal.setColor(QPalette.PlaceholderText, QColor(120, 120, 120))
pal.setColor(QPalette.Highlight, QColor(51, 153, 255))
pal.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
# 禁用态也压暗，避免灰白不可读
pal.setColor(QPalette.Disabled, QPalette.WindowText, QColor(140, 140, 140))
pal.setColor(QPalette.Disabled, QPalette.Text, QColor(140, 140, 140))
pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(140, 140, 140))
app.setPalette(pal)

# main.py 用 `from PyQt5.QtWidgets import *` 后会再调一次 QApplication(sys.argv)，
# 同时还会用 QApplication.setFont() 等静态方法。用代理对象：拦截「调用」复用
# 已建实例，其余属性（静态方法/类属性）全部转发给真正的 QApplication 类。
import PyQt5.QtWidgets as _W

_orig_qapp = _W.QApplication


class _QAppProxy(object):
    def __call__(self, *args, **kwargs):
        return _orig_qapp.instance() or _orig_qapp(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(_orig_qapp, name)


_W.QApplication = _QAppProxy()

# 以 __main__ 语义执行 main.py，使其 if __name__ == '__main__' 块照常触发
runpy.run_path("main.py", run_name="__main__")
