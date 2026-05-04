from enum import Enum

# 任务刺激类型枚举
class StimulationType(Enum):
    REWARD = "reward"
    PUNISHMENT = "punishment"  
    ENVIRONMENT = "environment"

# 任务左右轮刺激位置枚举
class StimulationPosition(Enum):
    LEFT = "left"
    RIGHT = "right"
    ALL = "all"

class StimulationSpec:
    """刺激规格类 - 之后可能根据不同的应用场景会进行拓展"""
    def __init__(self, stim_type, position):
        self.type = stim_type
        self.position = position
    
    def __str__(self):
        return f"{self.position.value}_{self.type.value}"
    
    def __repr__(self):
        return f"StimulationSpec({self.type}, {self.position})"
    
    
# 触发器枚举
class TriggerType(Enum):
    F1 = "f1"
    F2 = "f2"
    F3 = "f3"
    F4 = "f4"
    F5 = "f5"
    F6 = "f6"
    F7 = "f7"
    F8 = "f8"
    
# 刺激设置源枚举
class KeypressSource(Enum):
    KEYPRESS_F1 = "keypressf1"
    KEYPRESS_F2 = "keypressf2"
    KEYPRESS_F3 = "keypressf3"
    KEYPRESS_F4 = "keypressf4"
    KEYPRESS_F5 = "keypressf5"
    KEYPRESS_F6 = "keypressf6"
    KEYPRESS_F7 = "keypressf7"
    KEYPRESS_F8 = "keypressf8"
    
# 数字输出通道枚举
class DigitalOutput(Enum):
    DIGITAL_CHANNEL_01 = "DIGITAL-OUT-01"
    DIGITAL_CHANNEL_02 = "DIGITAL-OUT-02"
    
# 数字输入通道枚举
class DigitalIn(Enum):
    DIGITAL_CHANNEL_01 = "DIGITAL-IN-01"
    DIGITAL_CHANNEL_02 = "DIGITAL-IN-02"