"""
Maxwell 刺激单元池子化管理。

核心约束：闭环启动前所有 stim 电极的 routing + connect_electrode_to_stimulation
+ query unit 都由 RecordingMaxwell.connect 在 download **之前**完成，缓存
electrode → stim_unit 映射。本类在 download **之后** 只做 StimulationUnit
配置上电（power_up + connect + voltage_mode + dac_source）。

运行时切换激活子集走 StimulationUnit.connect(True/False) 通道
（mx.send(StimulationUnit(unit).connect(...))），禁止再次调用
array.connect_electrode_to_stimulation / disconnect_electrode_from_stimulation
（这两个是 download 前的 routing 配置 API，运行时不可用）。

每个 stim 电极对应一个 stim unit ID。本类负责跟踪映射、上电、运行时
开关状态机、退出时下电。
"""

from .errors import StimUnitError, check_send_ok


# Maxwell 硬件约束：单芯片最多 32 个刺激单元
MAX_STIM_UNITS = 32


class StimPool:
    """
    Stim 候选电极池 + stim unit 映射 + 激活状态机。

    Lifecycle
    ---------
    1. register_candidate(electrode_id) — 启动前注册所有可能用到的 stim 电极
    2. route_and_power_up(array)        — 一次性 route + 查询 unit + 上电
    3. activate(electrode_ids)           — 闭环运行时激活子集
    4. deactivate(electrode_ids)         — 闭环运行时停用子集
    5. cleanup()                          — 退出时下电所有 unit
    """

    def __init__(self, max_units=MAX_STIM_UNITS):
        if max_units > MAX_STIM_UNITS:
            raise StimUnitError(
                "max_units {} exceeds Maxwell hardware limit {}".format(max_units, MAX_STIM_UNITS)
            )
        self.max_units = max_units
        self._candidates = []           # list[int] 注册顺序保留
        self._electrode_to_unit = {}    # electrode_id -> stim_unit_id
        self._active_electrodes = set() # 当前已 connect 的电极集合
        self._array = None              # route 时记录，cleanup 时复用

    def register_candidate(self, electrode_id):
        """注册一个候选 stim 电极。必须在 route_and_power_up 之前调用。"""
        if self._array is not None:
            raise StimUnitError(
                "Cannot register candidates after route_and_power_up. "
                "Pool is already locked."
            )
        if electrode_id in self._candidates:
            return
        if len(self._candidates) >= self.max_units:
            raise StimUnitError(
                "Candidate pool full ({} units). Cannot add electrode {}.".format(
                    self.max_units, electrode_id
                )
            )
        self._candidates.append(electrode_id)

    def route_and_power_up(self, array, electrode_to_unit):
        """
        对 RecordingMaxwell 在 download 前已建立的 electrode → stim_unit
        映射做 StimulationUnit 配置上电。**本方法不调 connect_electrode_to_stimulation
        也不调 query_stimulation_at_electrode**，这两步已在 RecordingMaxwell.connect
        的 download 之前完成。

        必须在 RecordingMaxwell.connect 完成之后（即 download / wait /
        offset 全部就位）调用。

        Parameters
        ----------
        array : maxlab.chip.Array
            已经 route + connect_electrode_to_stimulation + download
            完成的 Array 对象。
        electrode_to_unit : dict[int, int]
            RecordingMaxwell._stim_electrode_to_unit 缓存的映射。

        Raises
        ------
        StimUnitError
            当映射缺失某个候选电极、或 power_up 命令失败时。
        """
        import maxlab as mx

        if self._array is not None:
            raise StimUnitError("StimPool.route_and_power_up called twice.")

        if not isinstance(electrode_to_unit, dict):
            raise StimUnitError(
                "route_and_power_up requires electrode_to_unit dict from RecordingMaxwell."
            )

        for electrode in self._candidates:
            if electrode not in electrode_to_unit:
                raise StimUnitError(
                    "Electrode {} missing from RecordingMaxwell._stim_electrode_to_unit; "
                    "stim_pool candidates and stim_electrodes must be consistent.".format(electrode)
                )
            unit_id = int(electrode_to_unit[electrode])
            self._electrode_to_unit[electrode] = unit_id

            print("[STIM_POOL] power_up unit={} electrode={} "
                  "(power_up(True) + connect(True) + voltage_mode + dac_source(0))".format(
                      unit_id, electrode
                  ))
            cmd = (mx.StimulationUnit(unit_id)
                   .power_up(True)
                   .connect(True)
                   .set_voltage_mode()
                   .dac_source(0))
            check_send_ok(
                mx.send(cmd),
                "stim unit {} power_up (electrode {})".format(unit_id, electrode),
            )

        # route_and_power_up 阶段所有 candidate unit 都 connect(True)，
        # 视为初始全部激活；后续 deactivate(...) 才把它们逐个关掉。
        self._active_electrodes = set(self._candidates)
        self._array = array
        print("[STIM_POOL] route_and_power_up done; init active set = {} "
              "(all candidates connect=True after power_up)".format(
                  sorted(self._active_electrodes)
              ))

    def get_unit(self, electrode_id):
        """返回某个候选电极对应的 stim unit ID。"""
        if electrode_id not in self._electrode_to_unit:
            raise StimUnitError(
                "Electrode {} is not in stim pool. Registered: {}".format(
                    electrode_id, list(self._electrode_to_unit.keys())
                )
            )
        return self._electrode_to_unit[electrode_id]

    def activate(self, electrode_ids):
        """
        运行时接通给定电极对应的 stim unit 输出（connect=True）。

        实现：mx.send(mx.StimulationUnit(unit_id).connect(True))。
        如果电极已经 active，幂等忽略。如果电极不在候选池中，抛异常。
        """
        import maxlab as mx

        if self._array is None:
            raise StimUnitError("StimPool not initialized. Call route_and_power_up first.")

        for electrode in electrode_ids:
            if electrode not in self._electrode_to_unit:
                raise StimUnitError(
                    "Cannot activate electrode {}: not in candidate pool.".format(electrode)
                )
            if electrode in self._active_electrodes:
                print("[STIM_POOL] activate skipped (already active) unit={} electrode={}".format(
                    self._electrode_to_unit[electrode], electrode
                ))
                continue
            unit_id = self._electrode_to_unit[electrode]
            print("[STIM_POOL] connect(True) unit={} electrode={} <-- HW toggle".format(
                unit_id, electrode
            ))
            check_send_ok(
                mx.send(mx.StimulationUnit(unit_id).connect(True)),
                "stim unit {} connect(True) (electrode {})".format(unit_id, electrode),
            )
            self._active_electrodes.add(electrode)

    def deactivate(self, electrode_ids):
        """
        运行时断开给定电极对应的 stim unit 输出（connect=False）。幂等。

        实现：mx.send(mx.StimulationUnit(unit_id).connect(False))。
        """
        import maxlab as mx

        if self._array is None:
            raise StimUnitError("StimPool not initialized.")

        for electrode in electrode_ids:
            if electrode not in self._active_electrodes:
                print("[STIM_POOL] deactivate skipped (already inactive) electrode={}".format(
                    electrode
                ))
                continue
            unit_id = self._electrode_to_unit[electrode]
            print("[STIM_POOL] connect(False) unit={} electrode={} <-- HW toggle".format(
                unit_id, electrode
            ))
            check_send_ok(
                mx.send(mx.StimulationUnit(unit_id).connect(False)),
                "stim unit {} connect(False) (electrode {})".format(unit_id, electrode),
            )
            self._active_electrodes.discard(electrode)

    def get_active_electrodes(self):
        """返回当前已激活的电极集合（拷贝）。"""
        return set(self._active_electrodes)

    def cleanup(self):
        """退出时下电所有 stim 单元。容错型，单步失败不影响后续清理。"""
        import maxlab as mx

        print("[STIM_POOL] cleanup: powering down {} units".format(
            len(self._electrode_to_unit)
        ))
        for electrode, unit_id in list(self._electrode_to_unit.items()):
            try:
                print("[STIM_POOL] cleanup: power_up(False) + connect(False) "
                      "unit={} electrode={}".format(unit_id, electrode))
                cmd = mx.StimulationUnit(unit_id).power_up(False).connect(False)
                mx.send(cmd)
            except Exception as exc:
                print("[STIM_POOL] cleanup: unit {} (electrode {}) power_down failed: {!r}".format(
                    unit_id, electrode, exc
                ))

        self._electrode_to_unit.clear()
        self._active_electrodes.clear()
        self._candidates.clear()
        self._array = None
