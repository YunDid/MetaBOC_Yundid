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
        self.record_electrodes = None     # 来自 cfg 解析；如显式注入则覆盖
        self.stim_electrodes = None       # 由上层在 connect 前注入，参与一次性 routing
        self._connected = False
        self._device_available = None  # 缓存设备探测结果，None 表示尚未探测
        self._array = None
        self._wells = None
        # download 前 connect_electrode_to_stimulation 完成后缓存的 electrode -> stim unit
        # 由 stim_pool 在 download 后做 StimulationUnit 上电时直接使用。
        self._stim_electrode_to_unit = {}

    def set_cfg_path(self, cfg_path):
        """
        在 connect() 之前注入 cfg 文件路径。MetaBOC 平台不实现活性扫描，
        cfg 由 MaxLab Live 完成后导出，平台只用 cfg_loader 解析其中的电极
        组并显式调用 select_electrodes，不走 Array.load_config 黑盒加载。
        """
        self.cfg_path = cfg_path

    def set_record_electrodes(self, electrodes):
        """
        显式注入记录电极列表。如未调用，connect() 会从 cfg 解析得到记录
        电极并使用。供调试 / 子集复跑使用。
        """
        self.record_electrodes = list(electrodes)

    def set_stim_electrodes(self, electrodes):
        """
        注入候选刺激电极列表。connect() 会在 download 前调用
        select_stimulation_electrodes，让 stim 电极一次性参与 routing。
        """
        self.stim_electrodes = list(electrodes) if electrodes else []

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
        执行 Maxwell 标准启动序列 + cfg 解析 + routing + stim 单元映射 + 校验。

        启动顺序（与 Maxwell 官方 closed_loop tutorial 与 Exp_code 实测路径
        对齐，必须严格遵守）：
          1. initialize_chip(wells)            # mx.initialize + enable_stimulation_power
                                               # + waitInit + activate(wells)
          2. 从 cfg 解析 record_electrodes      # 若未显式注入
          3. array = mx.Array("metaboc")
             array.reset()
             array.clear_selected_electrodes()
          4. array.select_electrodes(record_electrodes)
          5. 如有 stim_electrodes：array.select_stimulation_electrodes(stim_electrodes)
          6. array.route()
          7. **for each stim_el in stim_electrodes (download 之前)：**
                a. _has_routed_amplifier(stim_el) 校验
                b. array.connect_electrode_to_stimulation(stim_el)
                c. units = array.query_stimulation_at_electrode(stim_el)
                d. 缓存 self._stim_electrode_to_unit[stim_el] = unit
                e. 重复分配检查
          8. array.download(wells)
          9. wait_after_download + offset_calibration
          10. validate_record_electrodes 严格覆盖校验

        关键边界：connect_electrode_to_stimulation 与 query_stimulation_at_electrode
        必须在 download 之前调用 — 这是建立 stim 路由的步骤；download 之后再调
        会得到空 unit / Error 返回。stim_pool.route_and_power_up 在 download
        之后只做 StimulationUnit 上电，使用本方法缓存的 _stim_electrode_to_unit。
        """
        from . import session_lifecycle, cfg_loader

        if self.cfg_path is None:
            raise MxwserverError(
                "RecordingMaxwell.connect: cfg_path not set. Call set_cfg_path first."
            )

        import maxlab as mx

        print("[RECORDING] connect() entered; cfg_path={}".format(self.cfg_path))
        wells = session_lifecycle.initialize_chip()

        if not self.record_electrodes:
            print("[RECORDING] step: parse cfg → extract record electrodes")
            self.record_electrodes = cfg_loader.extract_electrodes(self.cfg_path)
            print("[RECORDING] parsed {} record electrodes from cfg".format(
                len(self.record_electrodes)
            ))

        print('[RECORDING] step: array = mx.Array("metaboc")')
        array = mx.Array("metaboc")
        print("[RECORDING] step: array.reset()")
        array.reset()
        print("[RECORDING] step: array.clear_selected_electrodes()")
        array.clear_selected_electrodes()
        print("[RECORDING] step: array.select_electrodes({} record electrodes)".format(
            len(self.record_electrodes)
        ))
        array.select_electrodes(self.record_electrodes)

        if self.stim_electrodes:
            print("[RECORDING] step: array.select_stimulation_electrodes({}) electrodes={}".format(
                len(self.stim_electrodes), self.stim_electrodes
            ))
            array.select_stimulation_electrodes(self.stim_electrodes)

        print("[RECORDING] step: array.route()")
        array.route()

        # download 前：建立 stim 电极 → stim unit 映射
        self._stim_electrode_to_unit = {}
        if self.stim_electrodes:
            print("[RECORDING] step: build stim electrode -> unit mapping (BEFORE download)")
            assigned_units = set()
            for stim_el in self.stim_electrodes:
                print("[RECORDING]   query_amplifier_at_electrode({})".format(stim_el))
                amp = array.query_amplifier_at_electrode(stim_el)
                if amp is None or (hasattr(amp, "__len__") and len(amp) == 0):
                    raise MxwserverError(
                        "stim electrode {} not routed to amplifier; cannot connect to stimulation. "
                        "Was it included in select_stimulation_electrodes / select_electrodes?".format(stim_el)
                    )

                print("[RECORDING]   connect_electrode_to_stimulation({}) <-- HW route".format(stim_el))
                array.connect_electrode_to_stimulation(stim_el)

                stim_units = array.query_stimulation_at_electrode(stim_el)
                if stim_units is None or (hasattr(stim_units, "__len__") and len(stim_units) == 0):
                    raise MxwserverError(
                        "No stim unit available for electrode {} after connect_electrode_to_stimulation.".format(stim_el)
                    )

                unit_id = int(stim_units) if not hasattr(stim_units, "__len__") else int(stim_units[0])
                if unit_id in assigned_units:
                    raise MxwserverError(
                        "Stim unit {} already assigned to a previous stim electrode. "
                        "Two stim electrodes mapped to same unit; pick a different electrode for {}.".format(
                            unit_id, stim_el
                        )
                    )
                assigned_units.add(unit_id)
                self._stim_electrode_to_unit[stim_el] = unit_id
                print("[RECORDING]   mapped electrode={} -> unit={}".format(stim_el, unit_id))
            print("[RECORDING] stim mapping complete: {}".format(
                self._stim_electrode_to_unit
            ))

        print("[RECORDING] step: array.download(wells={}) <-- HW download (commits route to chip)".format(wells))
        array.download(wells)
        print("[RECORDING] step: wait_after_download (mx.Timing.waitAfterDownload)")
        session_lifecycle.wait_after_download()
        print("[RECORDING] step: offset_calibration (mx.offset + waitInMX2Offset + clear_events)")
        session_lifecycle.offset_calibration()

        report = cfg_loader.validate_record_electrodes(array, self.record_electrodes)
        print("[RECORDING] electrode coverage: {}/{} record electrodes routed.".format(
            len(report["routed"]), len(self.record_electrodes)
        ))

        self._array = array
        self._wells = wells
        self._connected = True
        print("[RECORDING] connect() done; _connected=True")

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
