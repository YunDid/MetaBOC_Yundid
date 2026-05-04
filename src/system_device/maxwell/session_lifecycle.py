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

    通过发送一个无副作用的 system Event 命令做握手。如果 mxwserver
    未运行或不可达，maxlab 内部会抛出连接异常或返回非 "Ok" 字符串。

    Returns
    -------
    bool
        True 表示 mxwserver 可达。

    Raises
    ------
    MxwserverError
        当 mxwserver 不可达或返回失败状态。
    """
    import maxlab as mx

    try:
        result = mx.send(mx.system.Event(0))
    except Exception as exc:
        raise MxwserverError(
            "mxwserver probe raised exception: {!r}. Check whether mxwserver is running.".format(exc)
        )

    if result != "Ok":
        raise MxwserverError(
            "mxwserver probe returned {!r}. Server may be initializing or unreachable.".format(result)
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

    mx.initialize()

    result = mx.send(mx.Core().enable_stimulation_power(True))
    if result != "Ok":
        raise MxwserverError(
            "enable_stimulation_power failed: {!r}. The system did not initialize correctly.".format(result)
        )

    time.sleep(mx.Timing.waitInit)

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

    try:
        mx.offset()
    except Exception as exc:
        raise OffsetCalibrationError(
            "mx.offset() failed: {!r}".format(exc)
        )

    time.sleep(mx.Timing.waitInMX2Offset)
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

    if saving is not None:
        try:
            saving.stop_recording()
            time.sleep(mx.Timing.waitAfterRecording)
            saving.stop_file()
            saving.group_delete_all()
        except Exception as exc:
            print("Maxwell cleanup: saving teardown failed: {!r}".format(exc))

    if stim_pool is not None:
        try:
            stim_pool.cleanup()
        except Exception as exc:
            print("Maxwell cleanup: stim pool teardown failed: {!r}".format(exc))

    if array is not None:
        try:
            array.close()
        except Exception as exc:
            print("Maxwell cleanup: array.close() failed: {!r}".format(exc))
