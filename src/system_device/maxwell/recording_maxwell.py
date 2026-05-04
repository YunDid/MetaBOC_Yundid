"""
Maxwell Recording 角色 stub。

最小可跑实现：暴露与 MCS Recording / Intan RecordingIntan 等价的方法
签名，让 Communication 可以无差异调用。所有真实硬件交互通过 lazy
maxlab import + 占位返回值，等 C++ binary 工程完成后再补齐数据通路。

接口对齐目标：
- get_device_count() -> int                 设备发现
- set_record_para(sig)                       记录电极参数注入
- set_spike_detection_para(para)             spike 检测参数注入
- get_recording() -> (left_spike, right_spike)  闭环主入口，返回每通道 spike 数列表
"""

from .errors import MxwserverError


class RecordingMaxwell(object):
    """
    Maxwell 记录角色 stub。

    构造期不触碰 maxlab，保证 Windows 开发机能 import 通过。
    实际硬件交互延迟到 connect() 显式调用，由上层 MaxwellSystem 触发。
    """

    def __init__(self):
        self.recording_para = None
        self.spike_detection_para = None
        self.cfg_path = None
        self.record_electrodes = None
        self._connected = False
        self._device_available = None  # 缓存设备探测结果，None 表示尚未探测

    def set_cfg_path(self, cfg_path):
        """
        在 connect() 之前注入 cfg 文件路径。MetaBOC 平台不实现活性扫描，
        cfg 由 MaxLab Live 完成后导出，平台只做加载。
        """
        self.cfg_path = cfg_path

    def set_record_electrodes(self, electrodes):
        """注入期望参与记录的物理电极 ID 列表。供 cfg_loader 校验使用。"""
        self.record_electrodes = list(electrodes)

    def get_device_count(self):
        """
        设备发现入口。Maxwell 没有 USB 列表概念，等价语义是探测 mxwserver
        是否可达。返回 1 表示一个可用设备，0 表示不可用。

        本方法可在硬件不在场时调用，失败时返回 0 而非抛异常，与 MCS /
        Intan 的设备发现语义保持一致。
        """
        if self._device_available is not None:
            return 1 if self._device_available else 0

        try:
            from .session_lifecycle import check_mxwserver_alive
            check_mxwserver_alive()
            self._device_available = True
            print("Maxwell mxwserver is reachable.")
            return 1
        except MxwserverError as exc:
            self._device_available = False
            print("Maxwell device not found: {}".format(exc))
            return 0
        except Exception as exc:
            self._device_available = False
            print("Maxwell device probe failed (no maxlab installed?): {!r}".format(exc))
            return 0

    def connect(self):
        """
        执行 Maxwell 标准启动序列 + cfg 加载 + 电极校验。

        Phase B stub：仅完成框架调用，真实数据流接入留待 Phase D 的
        C++ binary 工程。
        """
        from . import session_lifecycle, cfg_loader

        if self.cfg_path is None:
            raise MxwserverError(
                "RecordingMaxwell.connect: cfg_path not set. Call set_cfg_path first."
            )

        import maxlab as mx

        wells = session_lifecycle.initialize_chip()

        array = mx.Array("metaboc")
        cfg_abs = cfg_loader.load_config(array, self.cfg_path)
        print("Maxwell loaded cfg: {}".format(cfg_abs))

        if self.record_electrodes:
            report = cfg_loader.validate_record_electrodes(array, self.record_electrodes)
            print("Maxwell electrode coverage: {}/{} routed.".format(
                len(report["routed"]), len(self.record_electrodes)
            ))

        array.route()
        array.download(wells)
        session_lifecycle.wait_after_download()
        session_lifecycle.offset_calibration()

        self._array = array
        self._wells = wells
        self._connected = True

    def set_record_para(self, sig):
        """与 MCS Recording 同名方法。注入记录电极配置对象。"""
        self.recording_para = sig

    def set_spike_detection_para(self, para):
        """与 MCS Recording 同名方法。注入 spike 检测参数。"""
        self.spike_detection_para = para

    def get_recording(self):
        """
        闭环主入口。返回 (left_spike, right_spike) 元组，每个元素是
        per-channel spike 数列表（与 MCS / Intan 一致）。

        Phase B stub：返回空列表占位。Phase D 完成后这里从 C++ binary
        喂过来的 spike 流中提取 per-electrode 计数。
        """
        if not self._connected:
            return [], []
        return [], []

    def stop_recording(self):
        """与 Intan recording 接口对齐。停止数据流。"""
        pass

    def disconnect(self):
        """断开 Maxwell 会话。容错型清理，保证硬件资源释放。"""
        if not self._connected:
            return
        from . import session_lifecycle
        session_lifecycle.cleanup_session(array=getattr(self, "_array", None))
        self._connected = False
