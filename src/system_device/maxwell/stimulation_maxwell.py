"""
Maxwell Stimulation 角色 stub。

最小可跑实现：暴露与 MCS Stimulation 等价的方法签名。所有刺激发放
通过 lazy maxlab + 占位日志，等 stim_pool 与编码器完整对接后再补齐
具体波形与时序。

接口对齐目标（Communication 直接调用）：
- set_recording(rec)                          关联 Recording 对象
- initial_device()                            设备初始化
- set_sti_signal(sig)                         注入刺激配置
- update_stimulation_left()                   左侧惩罚刺激（碰撞）
- update_stimulation_right()                  右侧惩罚刺激（碰撞）
- update_stimulation_left_right_reward()      双侧奖励
- update_record_stimulation(left, right)      环境编码主路径
- update_record_stimulation_dynamic_model_left(amp, dur)   MPC 路径左
- update_record_stimulation_dynamic_model_right(amp, dur)  MPC 路径右
"""

from .stim_pool import StimPool


class StimulationMaxwell(object):
    """
    Maxwell 刺激角色 stub。

    构造期不触碰 maxlab。stim_pool 在 connect() 阶段由 MaxwellSystem
    传入，本类只关心刺激语义（左/右/奖励/惩罚），不直接管理 unit 池。
    """

    def __init__(self):
        self.recording = None
        self.sti_sig = None
        self.stim_pool = None
        self._connected = False
        # 刺激电极的角色映射，例如 {"env_left": electrode_id, "env_right": electrode_id}
        # 由 channel_map_maxwell.npz 加载，stub 阶段留空
        self.role_to_electrode = {}

    def set_recording(self, recording):
        """与 MCS Stimulation 同名方法。建立刺激与记录对象的关联。"""
        self.recording = recording

    def initial_device(self):
        """与 MCS Stimulation 同名方法。Phase B stub：标记已初始化。"""
        self._connected = True
        print("Maxwell stimulation device marked as initialized (stub).")

    def set_sti_signal(self, sig):
        """与 MCS Stimulation 同名方法。注入刺激配置对象。"""
        self.sti_sig = sig

    def attach_stim_pool(self, stim_pool):
        """由 MaxwellSystem 在 connect() 完成后注入已 route 的 StimPool。"""
        if not isinstance(stim_pool, StimPool):
            raise TypeError("attach_stim_pool requires a StimPool instance.")
        self.stim_pool = stim_pool

    def set_role_mapping(self, role_to_electrode):
        """注入刺激电极角色映射。键如 'env_left' / 'env_right' / 'reward'。"""
        self.role_to_electrode = dict(role_to_electrode)

    def disconnect(self):
        """容错关闭。stim_pool 由 MaxwellSystem 统一 cleanup，本类不重复处理。"""
        self._connected = False

    # ---------- Communication 调用入口（环境编码主路径） ----------

    def update_record_stimulation(self, left, right):
        """
        环境编码主路径。MCS 实现会根据 left/right 频率发放刺激脉冲。
        Phase B stub：仅打印，待 stim_pool + DAC 序列实现后补齐。
        """
        if not self._connected:
            return
        print("Maxwell stim (env): left={} right={}".format(left, right))

    # ---------- 奖励 / 惩罚刺激 ----------

    def update_stimulation_left(self):
        """左侧惩罚刺激（碰撞触发）。"""
        if not self._connected:
            return
        print("Maxwell stim (punish-left)")

    def update_stimulation_right(self):
        """右侧惩罚刺激（碰撞触发）。"""
        if not self._connected:
            return
        print("Maxwell stim (punish-right)")

    def update_stimulation_left_right_reward(self):
        """双侧奖励刺激（无碰撞、ID 变化时触发）。"""
        if not self._connected:
            return
        print("Maxwell stim (reward both)")

    # ---------- MPC 动力学模型路径 ----------

    def update_record_stimulation_dynamic_model_left(self, ampli, duri):
        """MPC 左路径：振幅 + 脉宽序列直接发放。"""
        if not self._connected:
            return
        print("Maxwell stim (MPC-left): ampli={} duri={}".format(ampli, duri))

    def update_record_stimulation_dynamic_model_right(self, ampli, duri):
        """MPC 右路径：振幅 + 脉宽序列直接发放。"""
        if not self._connected:
            return
        print("Maxwell stim (MPC-right): ampli={} duri={}".format(ampli, duri))
