"""
Maxwell 记录电极「左右指派」对话框（Phase D / Stage 2）。

作用：从 cfg 解析出的记录电极里，让用户把电极分别勾选为「左轮」/「右轮」两组，
产出 recording_list = [left_electrodes, right_electrodes]。该列表经 main.py 在
设备切换时缓存、并在选定刺激参数时注入 RecordingMaxwell.recording_para，使
get_recording() 能按左右分别统计 spike（决定机器人按哪侧电极的放电转向）。

设计取舍：
- UI 全部在代码里构造（不依赖 .ui 文件），便于离屏测试与后续维护。
- 两侧均为多选列表，直接列出全部 cfg 电极（数量可能较多/冗余，按当前需求接受）。
- 这是 Stage 2 的轻量版；后续可打磨为图1 内嵌的电极选择器（见 Dashboard 代办）。

MaxOne HD-MEA 本身没有左/右半区，这里的「左右」是 MetaBOC 机器人左右轮的
传感来源指派，由用户按实验意图决定，不是芯片的物理属性。
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QPushButton, QAbstractItemView, QMessageBox,
)


class MaxwellRecordElectrodeDialog(QDialog):
    def __init__(self, electrodes, parent=None, preset_left=None, preset_right=None):
        super(MaxwellRecordElectrodeDialog, self).__init__(parent)
        self.setWindowTitle("Maxwell 记录电极 — 左右轮指派")
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.resize(520, 560)

        self._electrodes = [int(e) for e in electrodes]
        self._result = None  # accept → [left_ids, right_ids]；cancel → None

        root = QVBoxLayout(self)
        root.addWidget(QLabel(
            "从 cfg 的 {} 个记录电极中，分别勾选「左轮」「右轮」的记录电极。\n"
            "决定闭环里机器人按哪些电极的 spike 区分左右轮。可只选一侧；\n"
            "「跳过」则不分左右，get_recording 走全通道合并（机器人走直线）。"
            .format(len(self._electrodes))
        ))

        lists = QHBoxLayout()
        self.list_left = self._make_list(preset_left)
        self.list_right = self._make_list(preset_right)
        lists.addLayout(self._labeled("左轮  recording_list[0]", self.list_left))
        lists.addLayout(self._labeled("右轮  recording_list[1]", self.list_right))
        root.addLayout(lists)

        btns = QHBoxLayout()
        self.pb_ok = QPushButton("确定")
        self.pb_cancel = QPushButton("跳过 / 取消")
        self.pb_ok.clicked.connect(self._on_ok)
        self.pb_cancel.clicked.connect(self.reject)
        btns.addWidget(self.pb_ok)
        btns.addWidget(self.pb_cancel)
        root.addLayout(btns)

    def _make_list(self, preset):
        w = QListWidget()
        w.setSelectionMode(QAbstractItemView.MultiSelection)
        preset_set = set(int(x) for x in preset) if preset else set()
        for e in self._electrodes:
            item = QListWidgetItem(str(e))
            item.setData(Qt.UserRole, e)
            w.addItem(item)
            if e in preset_set:
                item.setSelected(True)
        return w

    def _labeled(self, title, widget):
        box = QVBoxLayout()
        box.addWidget(QLabel(title))
        box.addWidget(widget)
        return box

    def _selected_ids(self, w):
        return [w.item(i).data(Qt.UserRole)
                for i in range(w.count()) if w.item(i).isSelected()]

    def _on_ok(self):
        left = self._selected_ids(self.list_left)
        right = self._selected_ids(self.list_right)
        if not left and not right:
            QMessageBox.information(
                self, "未选择",
                "左右都未选。可点「跳过」走全通道合并 fallback，或至少选一侧。")
            return
        overlap = set(left) & set(right)
        if overlap:
            r = QMessageBox.question(
                self, "左右重叠",
                "电极 {} 同时在左右两组，确定继续？".format(sorted(overlap)),
                QMessageBox.Yes | QMessageBox.No)
            if r != QMessageBox.Yes:
                return
        self._result = [left, right]
        self.accept()

    def get_recording_list(self):
        """返回 [left_ids, right_ids]（int 列表）；用户取消/跳过则返回 None。"""
        return self._result
