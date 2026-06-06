"""
Maxwell 电极「左右轮指派」对话框（Phase D / Stage 2）。

一个对话框两用：
  - 记录电极指派（多选）：把记录电极分别勾为左轮/右轮 → recording_list=[left,right]，
    决定 get_recording 按哪些电极的 spike 区分左右。
  - 刺激电极指派（单选，single=True）：左轮/右轮各选一个刺激电极（当前范式固定 2 个）。

返回的 ID 是【字符串】（与 MCS 的 recording_list/stimulating_list 元素类型一致，
图2 SelectStimulatingPara 的 lineEdit 显示直接做字符串拼接）。下游
RecordingMaxwell._query_channel_for_electrode 调 query_amplifier_at_electrode 前会
int() 转回整数。

设计取舍：
- UI 全部代码内构（不依赖 .ui），便于离屏测试与维护。
- 直接列出全部 cfg 电极（数量可能较多/冗余，按当前需求接受）。
- MaxOne HD-MEA 没有左/右半区，这里的「左右」是 MetaBOC 机器人左右轮的角色指派，
  由用户按实验意图决定，不是芯片物理属性。
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QPushButton, QAbstractItemView, QMessageBox,
)


class MaxwellRecordElectrodeDialog(QDialog):
    def __init__(self, electrodes, parent=None, preset_left=None, preset_right=None,
                 single=False, title=None, intro=None, info_text=None):
        super(MaxwellRecordElectrodeDialog, self).__init__(parent)
        self._single = single
        self.setWindowTitle(title or ("Maxwell 刺激电极 — 左右轮" if single
                                      else "Maxwell 记录电极 — 左右轮指派"))
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.resize(520, 580)

        self._electrodes = [int(e) for e in electrodes]
        self._result = None  # accept → [left_ids(str), right_ids(str)]；cancel → None

        root = QVBoxLayout(self)

        if info_text:
            lbl_info = QLabel(info_text)
            lbl_info.setStyleSheet("color: rgb(40,90,160); font-weight: bold;")
            root.addWidget(lbl_info)

        if intro is None:
            if single:
                intro = ("从 cfg 的 {} 个电极中，给「左轮」「右轮」各选一个刺激电极。\n"
                         "两侧必须各选 1 个、且不能是同一个电极。"
                         .format(len(self._electrodes)))
            else:
                intro = ("从 cfg 的 {} 个记录电极中，分别勾选「左轮」「右轮」的记录电极。\n"
                         "决定闭环里机器人按哪些电极的 spike 区分左右轮。可只选一侧；\n"
                         "「跳过」则不分左右，get_recording 走全通道合并（机器人走直线）。"
                         .format(len(self._electrodes)))
        root.addWidget(QLabel(intro))

        lists = QHBoxLayout()
        self.list_left = self._make_list(preset_left)
        self.list_right = self._make_list(preset_right)
        lists.addLayout(self._labeled("左轮  [0]", self.list_left))
        lists.addLayout(self._labeled("右轮  [1]", self.list_right))
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
        w.setSelectionMode(QAbstractItemView.SingleSelection if self._single
                           else QAbstractItemView.MultiSelection)
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
        # 返回字符串 ID（与 MCS recording_list 元素类型一致）
        return [str(w.item(i).data(Qt.UserRole))
                for i in range(w.count()) if w.item(i).isSelected()]

    def _on_ok(self):
        left = self._selected_ids(self.list_left)
        right = self._selected_ids(self.list_right)
        if self._single:
            if len(left) != 1 or len(right) != 1:
                QMessageBox.information(self, "选择不完整", "刺激电极：左轮、右轮必须各选 1 个。")
                return
            if left[0] == right[0]:
                QMessageBox.information(self, "左右冲突", "左轮与右轮不能选同一个刺激电极。")
                return
        else:
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
        """返回 [left_ids, right_ids]（字符串列表）；取消/跳过则 None。"""
        return self._result
