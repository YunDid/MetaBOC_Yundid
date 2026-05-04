import enum

TASK = enum.Enum("TASK", ('Obstacle_Avoidance', 'Object_Tracking', 'Object_Grasping'))

MAP = enum.Enum("MAP", ('Empty_Map', 'Random_Map', 'Human_Map', 'Regular_Map'))

GRASP_STATE = enum.Enum("GRASP", ("Direction_Adjust", "Grasping", "Reset"))    # 抓取的三个阶段
GRASP_CONTROL = enum.Enum("CONTROL", ("Freedom", "Mode"))    # 第一种是全6个自由度的控制，第二种是离散状态控制

MAP_MODE = enum.Enum("MAP_MODE", ("Virtual", "Real"))

SYSTEM_DEVICE = enum.Enum("SYSTEM_DEVICE", ("MEA2100", "INTAN", "MAXWELL"))
