"""
Maxwell 接入层异常类型与错误检查辅助。

mxwserver 通过 mx.send() 返回字符串而非抛异常。本模块负责把
非 "Ok" 返回值转成 Python 异常，让上层 Communication 能用统一
的 try/except 风格处理失败。
"""


class MaxwellError(RuntimeError):
    """Maxwell 接入层所有异常的基类。"""


class MxwserverError(MaxwellError):
    """mxwserver 进程相关失败：未运行、不可达、握手失败。"""


class ConfigError(MaxwellError):
    """cfg 文件加载或电极配置校验失败。"""


class StimUnitError(MaxwellError):
    """刺激单元配置失败：路由失败、unit 重复分配、power_up 失败。"""


class OffsetCalibrationError(MaxwellError):
    """offset 校正阶段失败。"""


def check_send_ok(result, context):
    """
    校验 mx.send() 返回值。

    Maxwell Python API 的 send() 在失败路径上返回字符串 "Error" 或
    其他非 "Ok" 值，不抛异常。本函数把非 "Ok" 转成 MaxwellError。

    Parameters
    ----------
    result : str
        mx.send() 的返回值。
    context : str
        失败时附在异常信息里的上下文字符串，用于定位调用点。

    Raises
    ------
    MaxwellError
        当 result != "Ok" 时抛出。
    """
    if result != "Ok":
        raise MaxwellError(
            "Maxwell mx.send() failed at [{}]: returned {!r}".format(context, result)
        )
