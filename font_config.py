# -*- coding: utf-8 -*-
"""
完善的字体配置文件
用于设置UI的字体样式
"""

# 默认字体配置
DEFAULT_FONT_CONFIG = {
    "family": "Microsoft YaHei",  # 中文字体：微软雅黑
    "size": 10,                   # 字体大小
    "weight": 50,                 # 字体粗细
    "color": "#333333"            # 字体颜色
}

def get_global_font_style():
    """
    获取全局字体样式表
    英文使用Times New Roman，中文使用微软雅黑
    使用强选择器覆盖硬编码的字体设置
    """
    style_sheet = """
    /* 全局字体设置 - 使用强选择器覆盖硬编码字体 */
    QMainWindow, QWidget, QWidget * {
        font-family: "Microsoft YaHei", "Times New Roman", "SimSun", "Arial" !important;
        font-size: 10pt !important;
        font-weight: 50 !important;
        color: #333333 !important;
    }
    
    /* 基础控件字体 - 强制覆盖 */
    QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, 
    QGroupBox, QMenuBar, QStatusBar, QTextEdit, QLineEdit,
    QCheckBox, QRadioButton, QTabWidget, QTabBar, QToolBar,
    QToolButton, QMenu, QMenuItem, QAction, QSlider, QProgressBar {
        font-family: "Microsoft YaHei", "Times New Roman", "SimSun", "Arial" !important;
        font-size: 10pt !important;
        font-weight: 50 !important;
        color: #333333 !important;
    }
    
    /* 表格控件字体 */
    QTableWidget, QTableWidgetItem, QHeaderView, QTableView {
        font-family: "Microsoft YaHei", "Times New Roman", "SimSun", "Arial" !important;
        font-size: 9pt !important;
        font-weight: 50 !important;
        color: #333333 !important;
    }
    
    /* 列表控件字体 */
    QListWidget, QListWidgetItem, QTreeWidget, QTreeWidgetItem {
        font-family: "Microsoft YaHei", "Times New Roman", "SimSun", "Arial" !important;
        font-size: 9pt !important;
        font-weight: 50 !important;
        color: #333333 !important;
    }
    
    /* 滚动条字体 */
    QScrollBar, QScrollArea {
        font-family: "Microsoft YaHei", "Times New Roman", "SimSun", "Arial" !important;
        font-size: 8pt !important;
        font-weight: 50 !important;
        color: #333333 !important;
    }
    
    /* 对话框字体 */
    QDialog, QMessageBox, QInputDialog {
        font-family: "Microsoft YaHei", "Times New Roman", "SimSun", "Arial" !important;
        font-size: 10pt !important;
        font-weight: 50 !important;
        color: #333333 !important;
    }
    
    /* 特殊控件字体 */
    QCalendarWidget, QDateEdit, QTimeEdit, QDateTimeEdit {
        font-family: "Microsoft YaHei", "Times New Roman", "SimSun", "Arial" !important;
        font-size: 9pt !important;
        font-weight: 50 !important;
        color: #333333 !important;
    }
    
    /* 工具提示字体 */
    QToolTip {
        font-family: "Microsoft YaHei", "Times New Roman", "SimSun", "Arial" !important;
        font-size: 9pt !important;
        font-weight: 50 !important;
        color: #333333 !important;
        background-color: #FFFFDC;
        border: 1px solid #767676;
    }
    
    /* 状态栏字体 */
    QStatusBar {
        font-family: "Microsoft YaHei", "Times New Roman", "SimSun", "Arial" !important;
        font-size: 9pt !important;
        font-weight: 50 !important;
        color: #333333 !important;
    }
    
    /* 菜单栏字体 */
    QMenuBar {
        font-family: "Microsoft YaHei", "Times New Roman", "SimSun", "Arial" !important;
        font-size: 10pt !important;
        font-weight: 50 !important;
        color: #333333 !important;
    }
    
    /* 菜单项字体 */
    QMenu, QMenuItem {
        font-family: "Microsoft YaHei", "Times New Roman", "SimSun", "Arial" !important;
        font-size: 9pt !important;
        font-weight: 50 !important;
        color: #333333 !important;
    }
    
    /* 按钮字体 */
    QPushButton {
        font-family: "Microsoft YaHei", "Times New Roman", "SimSun", "Arial" !important;
        font-size: 10pt !important;
        font-weight: 50 !important;
        color: #333333 !important;
        padding: 5px;
    }
    
    /* 输入框字体 */
    QLineEdit, QTextEdit, QPlainTextEdit {
        font-family: "Microsoft YaHei", "Times New Roman", "SimSun", "Arial" !important;
        font-size: 10pt !important;
        font-weight: 50 !important;
        color: #333333 !important;
        padding: 2px;
    }
    
    /* 下拉框字体 */
    QComboBox {
        font-family: "Microsoft YaHei", "Times New Roman", "SimSun", "Arial" !important;
        font-size: 10pt !important;
        font-weight: 50 !important;
        color: #333333 !important;
        padding: 2px;
    }
    
    /* 数字输入框字体 */
    QSpinBox, QDoubleSpinBox {
        font-family: "Microsoft YaHei", "Times New Roman", "SimSun", "Arial" !important;
        font-size: 10pt !important;
        font-weight: 50 !important;
        color: #333333 !important;
        padding: 2px;
    }
    
    /* 标签字体 */
    QLabel {
        font-family: "Microsoft YaHei", "Times New Roman", "SimSun", "Arial" !important;
        font-size: 10pt !important;
        font-weight: 50 !important;
        color: #333333 !important;
    }
    
    /* 分组框字体 */
    QGroupBox {
        font-family: "Microsoft YaHei", "Times New Roman", "SimSun", "Arial" !important;
        font-size: 10pt !important;
        font-weight: 50 !important;
        color: #333333 !important;
        font-weight: bold !important;
    }
    
    /* 复选框和单选按钮字体 */
    QCheckBox, QRadioButton {
        font-family: "Microsoft YaHei", "Times New Roman", "SimSun", "Arial" !important;
        font-size: 10pt !important;
        font-weight: 50 !important;
        color: #333333 !important;
        spacing: 5px;
    }
    
    /* 覆盖所有可能的字体设置 */
    * {
        font-family: "Microsoft YaHei", "Times New Roman", "SimSun", "Arial" !important;
    }
    """
    
    return style_sheet

def create_global_font():
    """
    创建全局字体对象
    """
    from PyQt5.QtGui import QFont
    
    font = QFont()
    font.setFamily("Arial")  # 主要字体：微软雅黑
    # font.setFamily("Times New Roman")  # 主要字体：微软雅黑
    font.setPointSize(10)              # 字体大小：10点
    font.setWeight(50)                 # 字体粗细：正常
    font.setBold(False)                # 非粗体
    font.setItalic(False)              # 非斜体
    font.setUnderline(False)           # 非下划线
    font.setStrikeOut(False)           # 非删除线
    
    return font

def create_font_with_config(family="Microsoft YaHei", size=10, weight=50, bold=False, italic=False):
    """
    根据配置创建字体对象
    
    Args:
        family: 字体族
        size: 字体大小
        weight: 字体粗细
        bold: 是否粗体
        italic: 是否斜体
    
    Returns:
        QFont: 字体对象
    """
    from PyQt5.QtGui import QFont
    
    font = QFont()
    font.setFamily(family)
    font.setPointSize(size)
    font.setWeight(weight)
    font.setBold(bold)
    font.setItalic(italic)
    
    return font

def apply_font_to_widget(widget, font_family="Microsoft YaHei", size=10, weight=50):
# def apply_font_to_widget(widget, font_family="Times New Roman", size=12, weight=50):
    """
    将字体应用到指定控件及其所有子控件
    
    Args:
        widget: 目标控件
        font_family: 字体族
        size: 字体大小
        weight: 字体粗细
    """
    from PyQt5.QtGui import QFont
    
    # 创建字体对象
    font = QFont()
    font.setFamily(font_family)
    font.setPointSize(size)
    font.setWeight(weight)
    
    # 应用字体到控件
    widget.setFont(font)
    
    # 递归应用字体到所有子控件
    for child in widget.findChildren(QWidget):
        child.setFont(font)
