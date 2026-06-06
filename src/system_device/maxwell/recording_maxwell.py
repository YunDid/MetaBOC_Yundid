"""
Maxwell Recording 角色 stub。

暴露与 MCS Recording / Intan RecordingIntan 等价的方法签名，让 Communication
可以无差异调用。真实硬件交互通过 lazy maxlab import 延迟到 connect()；记录侧
spike 数据通路由框架外 C++ 探头 maxwell_streamer 经 stdout 喂入（Phase D）。

接口对齐目标：
- get_device_count() -> int                 设备发现
- set_record_para(sig)                       记录电极参数注入
- set_spike_detection_para(para)             spike 检测参数注入
- get_recording() -> (left_spike, right_spike)  闭环主入口，返回每通道 spike 数列表
"""

import os

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
        # download 之后 query_stimulation_at_electrode 拿到的 electrode -> stim unit
        # 权威映射。由 stim_pool 在 download 后做 StimulationUnit 上电时直接使用。
        self._stim_electrode_to_unit = {}
        # Phase D 记录侧：C++ 探头采集线程 + electrode→readout channel 惰性缓存
        self._recording_thread = None
        self._electrode_to_channel = {}
        self._lr_fallback_warned = False

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

        启动顺序（与 Maxwell 官方 closed_loop tutorial 与实测路径对齐）：
          1. initialize_chip(wells)            # mx.initialize + enable_stimulation_power
                                               # + waitInit + activate(wells)
          2. 从 cfg 解析 record_electrodes      # 若未显式注入
          3. array = mx.Array("metaboc")
             array.reset()
             array.clear_selected_electrodes()
          4. array.select_electrodes(record_electrodes)
          5. 如有 stim_electrodes：array.select_stimulation_electrodes(stim_electrodes)
          6. array.route()
          7. **for each stim_el in stim_electrodes (download 之前，路由声明)：**
                a. query_amplifier_at_electrode(stim_el) 校验已 routed
                b. array.connect_electrode_to_stimulation(stim_el)
          8. array.download(wells)
          9. wait_after_download + offset_calibration
          10. **for each stim_el in stim_electrodes (download 之后，硬件权威源)：**
                a. units = array.query_stimulation_at_electrode(stim_el)
                b. 缓存 self._stim_electrode_to_unit[stim_el] = unit
                c. 重复分配检查
          11. validate_record_electrodes 严格覆盖校验

        关键边界：query_stimulation_at_electrode 必须在 download **之后**调用。
        array.download() 会重新优化 routing，download 前 query 拿到的 unit_id 与
        download 后硬件实际生效的可能不同（1019 record + 2 stim 配置下实测
        right electrode unit 由 2 → 26 重新分配，commit aa1c870 的
        maxwell_connect_probe 在 P0 baseline 暴露此现象）。
        connect_electrode_to_stimulation 仍在 download 之前调（建立路由声明）。
        stim_pool.route_and_power_up 在 download 之后只做 StimulationUnit
        上电，使用本方法 download 后缓存的 _stim_electrode_to_unit（权威源）。
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

        # download 前：路由声明（amplifier 校验 + connect_electrode_to_stimulation）
        # query_stimulation_at_electrode 不在此处调 — array.download() 会重新优化 routing,
        # download 前 query 拿到的 unit_id 与 download 后实际生效的不一致（1019 record + 2 stim
        # 实测：right electrode 18884 由 unit 2 → unit 26 重新分配）。
        self._stim_electrode_to_unit = {}
        if self.stim_electrodes:
            print("[RECORDING] step: stim route declaration (BEFORE download): "
                  "amplifier check + connect_electrode_to_stimulation")
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

        print("[RECORDING] step: array.download(wells={}) <-- HW download (commits route to chip)".format(wells))
        array.download(wells)
        print("[RECORDING] step: wait_after_download (mx.Timing.waitAfterDownload)")
        session_lifecycle.wait_after_download()
        print("[RECORDING] step: offset_calibration (mx.offset + waitInMX2Offset + clear_events)")
        session_lifecycle.offset_calibration()

        # download 后：以硬件实际生效的 routing 为权威源建立 stim 电极 → unit 映射
        if self.stim_electrodes:
            print("[RECORDING] step: build stim electrode -> unit mapping "
                  "(AFTER download, ground truth)")
            assigned_units = set()
            for stim_el in self.stim_electrodes:
                stim_units = array.query_stimulation_at_electrode(stim_el)
                if stim_units is None or (hasattr(stim_units, "__len__") and len(stim_units) == 0):
                    raise MxwserverError(
                        "No stim unit assigned to electrode {} after download. "
                        "Was connect_electrode_to_stimulation called before download?".format(stim_el)
                    )

                # query_stimulation_at_electrode 实测返回字符串形式的 unit_id（如 '26'），
                # 不是 list/tuple。早期代码用 hasattr(__len__) 分支取 [0]，对字符串会
                # 切到首字符（'26'[0]='2' → int=2），在 unit_id ≥ 10 时悄无声息地解析错。
                # stimulate.html line 541 官方写法直接 int(stim)。这里只对 list/tuple
                # 显式判断，其他类型一律 int(...)，不走切片。
                if isinstance(stim_units, (list, tuple)):
                    unit_id = int(stim_units[0])
                else:
                    unit_id = int(stim_units)
                if unit_id in assigned_units:
                    raise MxwserverError(
                        "Stim unit {} already assigned to a previous stim electrode. "
                        "Two stim electrodes mapped to same unit; pick a different electrode for {}.".format(
                            unit_id, stim_el
                        )
                    )
                assigned_units.add(unit_id)
                self._stim_electrode_to_unit[stim_el] = unit_id
                print("[RECORDING]   mapped electrode={} -> unit={} (post-download)".format(stim_el, unit_id))
            print("[RECORDING] stim mapping complete (post-download): {}".format(
                self._stim_electrode_to_unit
            ))

        report = cfg_loader.validate_record_electrodes(array, self.record_electrodes)
        print("[RECORDING] electrode coverage: {}/{} record electrodes routed.".format(
            len(report["routed"]), len(self.record_electrodes)
        ))

        self._array = array
        self._wells = wells
        self._connected = True

        # Phase D：在 connect 尾部启动框架外 C++ 只读探头，把 server 侧检测的 spike
        # 经 stdout 喂进来。
        # 流的来源：mxwserver 启动 + 芯片在位即出流，与 Python 无关（2026-06-06 真机插片
        # 实测：未跑任何 Python 设置、仅 server+芯片，探头即刷 S/H）。本处 route/download
        # 不是“让流存在”的前提，而是把我们选的记录/刺激电极映射到 readout 通道——这样
        # get_recording 的 electrode→channel 才对得上我们的配置。流也不依赖
        # mx.Saving()/start_recording（不写盘照样有流）。
        self._start_recording_thread()

        print("[RECORDING] connect() done; _connected=True")

    def set_record_para(self, sig):
        """与 MCS Recording 同名方法。注入记录电极配置对象。"""
        self.recording_para = sig

    def set_spike_detection_para(self, para):
        """与 MCS Recording 同名方法。注入 spike 检测参数。"""
        self.spike_detection_para = para

    def get_recording(self):
        """
        闭环主入口。返回 (left_spike, right_spike)，每个元素是 per-channel spike 数
        列表（与 MCS / Intan 契约一致，供下游 np.sum / np.array(...).mean(axis=0) /
        save_spike_reference 的 cur[0] 使用，故必须是非空、长度稳定的序列）。

        Phase D：从后台 C++ 探头线程的滚动窗口取每通道计数。
        - 默认“合并模式”：每侧返回长度 1 的 [该侧总计数]（对齐 MCS 主闭环最省事）。
          逐电极模式（喂 AI 动力学模型）只需把下面 total 换成 per-channel 列表即可，
          上层契约不变，留待要跑 AI 时再切。
        - 左右分组来源：recording_para.recording_list[0]/[1]（方案 A，沿用 MCS/Intan
          注入口）。recording_list 尚未注入时（Maxwell 图1 GUI 适配未完成），退化为
          “两侧都返回窗口内全通道总计数”的安全占位，并一次性告警。
        """
        if not self._connected or self._recording_thread is None:
            return [0], [0]  # 非连接态：非空 length-1 占位，满足下游 cur[0]/mean 契约

        left_ch, right_ch = self._resolve_left_right_channels()
        if left_ch is None:
            # recording_list 未注入 → 无法分左右，返回全通道总计数作占位
            if not self._lr_fallback_warned:
                print("[RECORDING] WARN: recording_para.recording_list 未注入，"
                      "get_recording 暂返回全通道合并计数（左右相同占位）；"
                      "待 Maxwell 图1 GUI 适配注入左右记录电极后自动启用分组。")
                self._lr_fallback_warned = True
            total = self._recording_thread.get_total_count(None)
            return [total], [total]

        left_total = self._recording_thread.get_total_count(left_ch)
        right_total = self._recording_thread.get_total_count(right_ch)
        return [left_total], [right_total]

    # ------------------------------------------------------- Phase D 内部辅助
    def _streamer_binary_path(self):
        """框架外 C++ 探头编译产物路径（仓库根/cpp/maxwell_streamer/build/maxwell_streamer）。"""
        here = os.path.dirname(os.path.abspath(__file__))
        # 本文件目录 = <repo>/src/system_device/maxwell → 上溯 3 层到 <repo> 根
        repo_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
        return os.path.join(repo_root, "cpp", "maxwell_streamer", "build", "maxwell_streamer")

    def _start_recording_thread(self):
        """connect 尾部启动 C++ 探头采集线程。"""
        from .recording_maxwell_thread import ReadMaxwellDataThread
        binary = self._streamer_binary_path()
        # MaxOne 固定 well 0；MaxTwo 多 well 时由 self._wells 推导（此处接 MaxOne）
        well = 0
        self._recording_thread = ReadMaxwellDataThread(binary_path=binary, well=well)
        self._recording_thread.start_streamer()
        print("[RECORDING] maxwell_streamer 探头线程已启动: {}".format(binary))

    def _query_channel_for_electrode(self, electrode):
        """electrode → 数据流 readout channel（即 S 行里的 <channel>）。

        用 array.query_amplifier_at_electrode 取该电极路由到的 amplifier；防字符串
        切片陷阱（'26'[0] → '2'），list/tuple 取 [0]，其余一律 int(...)。

        ⚠ 真机必验③：本方法假定 query_amplifier_at_electrode 返回的 amplifier 编号
        与 SpikeEvent.channel（DataStreamerFiltered 给的 0–1023 readout 通道）同一套
        编号。这点文档未给死保证，必须在真机用“已知有放电的单电极”对一次再定论。
        """
        if self._array is None:
            return None
        try:
            amp = self._array.query_amplifier_at_electrode(electrode)
        except Exception as exc:
            print("[RECORDING] query_amplifier_at_electrode({}) 失败: {!r}".format(electrode, exc))
            return None
        if amp is None:
            return None
        if isinstance(amp, (list, tuple)):
            if len(amp) == 0:
                return None
            return int(amp[0])
        try:
            return int(amp)
        except (ValueError, TypeError):
            return None

    def _ensure_electrode_channel_map(self, electrodes):
        """惰性补建 electrode → channel 映射（只查未缓存过的电极）。"""
        for e in electrodes:
            if e in self._electrode_to_channel:
                continue
            ch = self._query_channel_for_electrode(e)
            if ch is not None:
                self._electrode_to_channel[e] = ch

    def _resolve_left_right_channels(self):
        """从 recording_para.recording_list 推出 (left_channels, right_channels)。

        无法解析（recording_para/recording_list 未就绪）时返回 (None, None)。
        """
        para = self.recording_para
        if para is None or not hasattr(para, "recording_list"):
            return None, None
        rl = para.recording_list
        if not rl or len(rl) < 2:
            return None, None
        left_els = list(rl[0]) if rl[0] else []
        right_els = list(rl[1]) if rl[1] else []
        if not left_els and not right_els:
            return None, None
        self._ensure_electrode_channel_map(left_els + right_els)
        left_ch = [self._electrode_to_channel[e] for e in left_els if e in self._electrode_to_channel]
        right_ch = [self._electrode_to_channel[e] for e in right_els if e in self._electrode_to_channel]
        return left_ch, right_ch

    def stop_recording(self):
        """与 Intan recording 接口对齐。停止 C++ 探头数据流。"""
        if self._recording_thread is not None:
            self._recording_thread.stop_streamer()
            self._recording_thread = None

    def disconnect(self):
        """断开 Maxwell 会话。先停探头子进程，再容错清理硬件资源。"""
        if not self._connected:
            return
        self.stop_recording()  # 先停 C++ 探头，避免 server close 后探头仍在读
        from . import session_lifecycle
        session_lifecycle.cleanup_session(array=getattr(self, "_array", None))
        self._connected = False
