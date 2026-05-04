"""
Maxwell 系统包装类。

参照 INTAN_System 模式，把 RecordingMaxwell + StimulationMaxwell 组合
为单一系统对象，供 Communication 通过 self.maxwell_sys 访问。

生命周期：
- __init__: 仅实例化角色对象，不触碰 maxlab
- start_session(cfg_path, record_electrodes, stim_electrodes):
    完整启动序列 → 8 步初始化 → cfg 加载 → stim pool route 与上电
- stop_connect(): 容错清理所有资源
"""

from src.system_device.maxwell.recording_maxwell import RecordingMaxwell
from src.system_device.maxwell.stimulation_maxwell import StimulationMaxwell
from src.system_device.maxwell.stim_pool import StimPool


class MaxwellSystem(object):
    """与 INTAN_System 等价的设备包装类。"""

    def __init__(self):
        self.recording = RecordingMaxwell()
        self.stimulating = StimulationMaxwell()
        self.stim_pool = None

    def start_session(self, cfg_path, record_electrodes=None, stim_electrodes=None,
                      role_mapping=None):
        """
        完整启动 Maxwell 会话。

        Parameters
        ----------
        cfg_path : str or Path
            MaxLab Live 导出的 .cfg 文件路径。
        record_electrodes : list[int] or None
            参与记录的物理电极 ID。校验 cfg 是否覆盖。
        stim_electrodes : list[int] or None
            候选刺激电极池（≤32）。一次性 route + 上电。
        role_mapping : dict or None
            刺激电极角色映射，注入到 StimulationMaxwell。
            如 {"env_left": 15000, "env_right": 15500}
        """
        if record_electrodes is not None:
            self.recording.set_record_electrodes(record_electrodes)
        self.recording.set_cfg_path(cfg_path)

        # Step 1：cfg 加载 + 8 步初始化（在 RecordingMaxwell.connect 内部完成）
        self.recording.connect()

        # Step 2：stim pool 注册 + route + 上电（如有 stim 电极）
        if stim_electrodes:
            self.stim_pool = StimPool()
            for electrode in stim_electrodes:
                self.stim_pool.register_candidate(electrode)
            # 注意：select_stimulation_electrodes 必须在 array.route() 之前完成。
            # Phase B stub 阶段不重新 route（cfg 中可能已含 stim 电极配置），
            # 此处仅做 connect_electrode_to_stimulation + 查询 unit + 上电。
            # Phase B 后续真机验证时如需重 route，在此扩展 array.select_stimulation_electrodes 调用。
            self.stim_pool.route_and_power_up(self.recording._array)
            self.stimulating.attach_stim_pool(self.stim_pool)

        if role_mapping:
            self.stimulating.set_role_mapping(role_mapping)

        # Step 3：标记 stimulation 已就绪
        self.stimulating.initial_device()

    def stop_connect(self):
        """与 INTAN_System.stop_connect 等价。容错清理顺序：stim → recording。"""
        from src.system_device.maxwell import session_lifecycle

        try:
            self.stimulating.disconnect()
        except Exception as exc:
            print("MaxwellSystem stop_connect: stimulating teardown failed: {!r}".format(exc))

        try:
            session_lifecycle.cleanup_session(
                array=getattr(self.recording, "_array", None),
                stim_pool=self.stim_pool,
            )
        except Exception as exc:
            print("MaxwellSystem stop_connect: session cleanup failed: {!r}".format(exc))

        try:
            self.recording.disconnect()
        except Exception as exc:
            print("MaxwellSystem stop_connect: recording teardown failed: {!r}".format(exc))

        self.stim_pool = None
