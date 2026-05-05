"""
Maxwell 会话生命周期：mxwserver 探活、芯片初始化、offset 校正、清理。

启动序列严格遵循 Maxwell Python API tutorial 的 8 步流程，禁止省略
任何一步、禁止用任意 sleep 替代官方 mx.Timing.xxx 常量。
"""

import time

from .errors import (
    MxwserverError,
    OffsetCalibrationError,
    check_send_ok,
)


# MaxOne 默认只有一个 well，固定为 [0]。MaxTwo 可指定多个 well。
DEFAULT_WELLS = [0]


def check_mxwserver_alive():
    """
    探测 mxwserver 是否运行可达。

    使用 maxlab.query_DAC_lsb_mV() 作为探活调用：无参数、无副作用，
    只查询 DAC 硬件常数。该函数内部会与 mxwserver 通信，server 不可达
    时抛异常，可达时返回非空字符串。

    禁止用 mx.system.Event 探活——Event 真实签名是
    (well_id, event_type, user_id, properties)，会触发 status_out 事件
    并记入数据流，对实验状态有侵入。

    Returns
    -------
    bool
        True 表示 mxwserver 可达。

    Raises
    ------
    MxwserverError
        当 mxwserver 不可达或返回空值。
    """
    import maxlab as mx

    try:
        result = mx.query_DAC_lsb_mV()
    except Exception as exc:
        raise MxwserverError(
            "mxwserver probe (query_DAC_lsb_mV) raised: {!r}. "
            "Check whether mxwserver is running and reachable.".format(exc)
        )

    if not result:
        raise MxwserverError(
            "mxwserver probe (query_DAC_lsb_mV) returned empty value: {!r}. "
            "Server may be initializing or in degraded state.".format(result)
        )
    return True


def initialize_chip(wells=None):
    """
    标准启动序列 Step 1-4：芯片复位 + 开启刺激电源 + 等待初始化 + 激活 well。

    Parameters
    ----------
    wells : list[int] or None
        要激活的 well 编号列表。MaxOne 用 [0]（默认），MaxTwo 按实验需要传入。

    Returns
    -------
    list[int]
        实际激活的 wells 列表，供后续 download / saving 复用。

    Raises
    ------
    MxwserverError
        当 enable_stimulation_power 失败时。
    """
    import maxlab as mx

    if wells is None:
        wells = list(DEFAULT_WELLS)

    print("[SESSION] initialize_chip: wells={}".format(wells))
    print("[SESSION]   mx.initialize() <-- HW reset")
    mx.initialize()

    print("[SESSION]   mx.send(Core.enable_stimulation_power(True))")
    result = mx.send(mx.Core().enable_stimulation_power(True))
    if (result or "").upper() != "OK":
        raise MxwserverError(
            "enable_stimulation_power failed: {!r}. The system did not initialize correctly.".format(result)
        )

    print("[SESSION]   time.sleep(mx.Timing.waitInit={}s)".format(mx.Timing.waitInit))
    time.sleep(mx.Timing.waitInit)

    print("[SESSION]   mx.activate(wells={})".format(wells))
    mx.activate(wells)
    return wells


def offset_calibration():
    """
    标准启动序列 Step 7-8：offset 补偿 + 等待 + 清空事件缓冲。

    必须在 array.download() 完成且 mx.Timing.waitAfterDownload 等待
    结束之后调用。每次电极配置变化（重新 download）后必须重做 offset。

    Raises
    ------
    OffsetCalibrationError
        当底层抛出异常时（mx.offset 本身在文档中无明确返回值约定）。
    """
    import maxlab as mx

    print("[SESSION] offset_calibration: mx.offset() <-- HW calibration")
    try:
        mx.offset()
    except Exception as exc:
        raise OffsetCalibrationError(
            "mx.offset() failed: {!r}".format(exc)
        )

    print("[SESSION]   time.sleep(mx.Timing.waitInMX2Offset={}s)".format(mx.Timing.waitInMX2Offset))
    time.sleep(mx.Timing.waitInMX2Offset)
    print("[SESSION]   mx.clear_events() (one-shot post-offset; will be re-cleared once after stim_pool route_and_power_up)")
    mx.clear_events()


def wait_after_download():
    """
    标准启动序列 Step 6：array.download() 之后的强制等待。

    封装为函数让上层调用点不直接依赖 mx.Timing.waitAfterDownload 常量名，
    便于未来 SDK 升级时统一调整。
    """
    import maxlab as mx

    time.sleep(mx.Timing.waitAfterDownload)


def cleanup_session(array=None, stim_pool=None, saving=None):
    """
    会话清理：关停 saving、断开 stim 单元、关闭 array、释放资源。

    设计为容错型：每个清理步骤独立 try/except，单步失败不影响后续清理，
    保证即使部分组件已经异常仍能尽量释放硬件资源。

    Parameters
    ----------
    array : maxlab.chip.Array or None
        要关闭的电极阵列对象。
    stim_pool : StimPool or None
        要清理的刺激单元池。
    saving : maxlab.Saving or None
        正在录制的 Saving 对象。
    """
    import maxlab as mx

    print("[SESSION] cleanup_session: array={}, stim_pool={}, saving={}".format(
        "yes" if array is not None else "no",
        "yes" if stim_pool is not None else "no",
        "yes" if saving is not None else "no",
    ))

    if saving is not None:
        try:
            print("[SESSION]   saving.stop_recording / stop_file / group_delete_all")
            saving.stop_recording()
            time.sleep(mx.Timing.waitAfterRecording)
            saving.stop_file()
            saving.group_delete_all()
        except Exception as exc:
            print("[SESSION] cleanup: saving teardown failed: {!r}".format(exc))

    if stim_pool is not None:
        try:
            print("[SESSION]   stim_pool.cleanup() (delegates to StimPool.cleanup)")
            stim_pool.cleanup()
        except Exception as exc:
            print("[SESSION] cleanup: stim pool teardown failed: {!r}".format(exc))

    if array is not None:
        try:
            print("[SESSION]   array.close() <-- HW")
            array.close()
        except Exception as exc:
            print("[SESSION] cleanup: array.close() failed: {!r}".format(exc))
