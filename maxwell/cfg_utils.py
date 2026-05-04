"""
cfg_utils.py
------------
Maxwell .cfg 配置文件解析 + 刺激单元冲突自动解决工具。

依赖：maxlab（仅在线环境中可用，离线分析时仅使用 parse 功能）
"""

import re
from typing import List, Dict, Tuple, Optional


# ==================== CFG 解析 ====================

def parse_cfg(cfg_path: str) -> List[Dict]:
    """解析 Maxwell Scope GUI 导出的 .cfg 文件。

    CFG 格式：分号分隔条目，每条为 `channel(electrode)x/y`
    例：0(6670)1225/525;2(7112)1260/560;...

    Returns
    -------
    List[Dict]
        每个元素: {"channel": int, "electrode": int, "x": float, "y": float}
    """
    with open(cfg_path, "r") as f:
        raw = f.read().strip()

    pattern = re.compile(r"(\d+)\((\d+)\)([\d.]+)/([\d.]+)")
    entries = []
    for m in pattern.finditer(raw):
        entries.append({
            "channel": int(m.group(1)),
            "electrode": int(m.group(2)),
            "x": float(m.group(3)),
            "y": float(m.group(4)),
        })
    return entries


def extract_electrodes(cfg_path: str) -> List[int]:
    """从 .cfg 文件提取所有电极编号（去重，保持顺序）。

    Returns
    -------
    List[int]
        电极编号列表
    """
    entries = parse_cfg(cfg_path)
    seen = set()
    electrodes = []
    for e in entries:
        if e["electrode"] not in seen:
            seen.add(e["electrode"])
            electrodes.append(e["electrode"])
    return electrodes


# ==================== 刺激单元冲突解决 ====================

def connect_stim_electrodes_with_fallback(
    stim_electrodes: List[int],
    array,  # mx.Array
    max_search_radius: int = 5,
) -> Tuple[List[int], List[int], Dict[int, int]]:
    """将刺激电极连接到独立的刺激单元，冲突时自动搜索邻近电极替代。

    算法：
    1. 依次尝试将每个刺激电极连接到刺激单元
    2. 若连接失败（无可用刺激通道）或与已占用的刺激单元冲突，
       则通过 maxlab.util.electrode_neighbors() 逐半径扩大搜索邻近电极
    3. 对每个邻近候选电极尝试连接，直到找到一个独占的刺激单元
    4. 搜索到 max_search_radius 仍无解则报错

    Parameters
    ----------
    stim_electrodes : List[int]
        期望的刺激电极列表（≤32）
    array : mx.Array
        已完成 route() 的 Array 对象
    max_search_radius : int
        邻近搜索最大半径（逻辑欧氏距离），默认 5

    Returns
    -------
    final_electrodes : List[int]
        实际连接的电极列表（与输入等长，被替换的位置为邻近电极）
    stim_units : List[int]
        对应的刺激单元编号
    substitutions : Dict[int, int]
        被替换的映射 {原始电极: 替代电极}，无替换则为空 dict
    """
    import maxlab as mx

    occupied_units: set = set()
    used_electrodes: set = set()
    final_electrodes: List[int] = []
    stim_units: List[int] = []
    substitutions: Dict[int, int] = {}

    for original_el in stim_electrodes:
        candidates = [original_el]

        # 预生成各半径的邻近电极作为候选
        for radius in range(1, max_search_radius + 1):
            neighbors = mx.util.electrode_neighbors(original_el, radius)
            for n in neighbors:
                if n not in candidates:
                    candidates.append(n)

        connected = False
        for candidate in candidates:
            if candidate in used_electrodes:
                continue

            array.connect_electrode_to_stimulation(candidate)
            stim_str = array.query_stimulation_at_electrode(candidate)

            if len(stim_str) == 0:
                # 该电极无法连接到任何刺激通道，跳过
                continue

            stim_unit = int(stim_str)
            if stim_unit in occupied_units:
                # 冲突：此刺激单元已被占用，尝试下一个候选
                continue

            # 成功：独占的刺激单元
            occupied_units.add(stim_unit)
            used_electrodes.add(candidate)
            final_electrodes.append(candidate)
            stim_units.append(stim_unit)

            if candidate != original_el:
                substitutions[original_el] = candidate
                print(f"[FALLBACK] electrode {original_el} -> {candidate} (stim_unit {stim_unit})")
            else:
                print(f"[OK] electrode {original_el} -> stim_unit {stim_unit}")

            connected = True
            break

        if not connected:
            raise RuntimeError(
                f"无法为电极 {original_el} 找到独占的刺激单元，"
                f"已搜索半径 {max_search_radius} 内的所有邻近电极"
            )

    return final_electrodes, stim_units, substitutions


# ==================== 便捷入口 ====================

if __name__ == "__main__":
    import sys
    # 读取当前目录下的 15h30m39s.cfg
    cfg_path = "H:\\MetaBOC_Yundid\\maxwell\\15h30m39s.cfg"
    entries = parse_cfg(cfg_path)
    electrodes = extract_electrodes(cfg_path)

    print(f"\n=== CFG 解析结果: {cfg_path} ===")
    print(f"总条目数: {len(entries)}")
    print(f"去重电极数: {len(electrodes)}")
    
    print(f"\n前 10 条条目:")
    for e in entries[:10]:
        print(f"  channel={e['channel']:>3d}  electrode={e['electrode']:>5d}  "
              f"pos=({e['x']}, {e['y']})")
    print(f"\n全部电极列表 (共 {len(electrodes)} 个):")
    print(electrodes)
