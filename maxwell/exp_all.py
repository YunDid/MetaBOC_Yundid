#!/usr/bin/python

"""
STIMULATION EXAMPLES

This script can be used to run stimulation on a MaxTwo or
MaxOne system via the Python API. Several combinations of pulse
trains can be run and the script can be called from the command
line directly. 

Warning: No recordings are made in this example, but you can find
an example about how to record in another script, called `recordings.py`.
"""

import time

import maxlab as mx

from typing import List, Optional
from pathlib import Path

# =============================================== SystemAPI ===================================================

def initialize_system() -> None:
    """Initialize system into a defined state

    The function initializes the system into a defined state before
    starting any script. This way, one can be sure that the system
    is always in the same state while running the script, regardless
    of what has been done before with it. The function also powers on
    the stimulation units, which are turned off by default.

    Raises
    ------
    RuntimeError
        If system does not initialize correctly.

    """
    mx.initialize()
    if mx.send(mx.Core().enable_stimulation_power(True)) != "Ok":
        raise RuntimeError("The system didn't initialize correctly.")


def configure_array(electrodes: List[int], stim_electrodes: List[int]) -> mx.Array:
    """Configure array

    This function configures the array, given a list of recording
    and stimulation electrodes. An important step in this function
    is the routing, which happens once the electrodes are selected.

    Parameters
    ----------
    electrodes : List[int]
        List of the index of the recording electrodes
    stim_electrodes : List[int]
        List of the index of the stimulation electrodes

    Returns
    -------
    mx.Array
        The configured array.

    Notes
    -----
    To finalize the configuration, the array needs to be downloaded to
    the MaxOne/Two. However, before being able to do that, we first need
    to connect the stimulation units to the stimulation electrodes with the
    function `connect_stim_units_to_stim_electrodes`. This is the reason
    why `array.download` is not part of this function.

    """
    array = mx.Array("stimulation")
    array.reset()
    array.clear_selected_electrodes()
    array.select_electrodes(electrodes)
    array.select_stimulation_electrodes(stim_electrodes)
    array.route()
    return array


def load_config(config_file: str) -> mx.Array:
    """Load a previously created configuration

    Load and configure an array from a previously generated
    configuration, either through the Python API or through
    the Scope GUI.

    Parameters
    ----------
    config_file : str
        Name of the previously generated config file.

    Returns
    -------
    mx.Array
        The configured array.

    Raises
    ------
    FileNotFoundError
        If the file cannot be found.
    Exception
        If the config_file cannot be loaded properly.

    """
    path = Path(config_file)
    if not path.is_file():
        raise FileNotFoundError(f"Config file '{config_file}' not found.")
    array = mx.Array("stimulation")
    try:
        array.load_config(config_file)
    except Exception as e:
        raise Exception(f"Error loading config file '{config_file}': {str(e)}")
    return array


def connect_stim_units_to_stim_electrodes(
    stim_electrodes: List[int], array: mx.Array
) -> List[int]:
    """Connect the stimulation units to the stimulation electrodes

    Once an array configuration has been obtained, either through routing
    or through loading a previous configuration, the stimulation units
    can be connected to the desired electrodes.

    Notes
    -----
    With this step, one needs to be careful, in rare cases it can happen
    that an electrode cannot be stimulated. For example, the electrode
    could not be routed (due to routing constraints), and the error message
    "No stimulation channel can connect to electrode: ..." will be printed.
    If this situation occurs, it is recommended to then select the electrode
    next to it.

    Parameters
    ----------
    stim_electrodes : List[int]
        List of the index of the stimulation electrodes
    array : mx.Array
        The configured array

    Returns
    -------
    List[str]
        List of stimulation units indices corresponding to the connected
        stimulation electrodes

    Raises
    ------
    RuntimeError
        If an electrode cannot be connected to a stimulation unit.
        If two electrodes are connected to the same stimulation unit.
    """
    stim_units: List[int] = []
    for stim_el in stim_electrodes:
        array.connect_electrode_to_stimulation(stim_el)
        stim = array.query_stimulation_at_electrode(stim_el)
        if len(stim) == 0:
            raise RuntimeError(
                f"No stimulation channel can connect to electrode: {str(stim_el)}"
            )
        stim_unit_int = int(stim)
        if stim_unit_int in stim_units:
            raise RuntimeError(
                f"Two electrodes connected to the same stim unit.\
                               This is not allowed. Please Select a neighboring electrode of {stim_el}!"
            )
        else:
            stim_units.append(stim_unit_int)
    return stim_units


def powerup_stim_unit(stim_unit: int) -> mx.StimulationUnit:
    """Power up and connect a stimulation unit

    This function powers up and connect a specific stimulation
    unit in the MaxOne/Two. Without it, we would not be able to
    send the prepared sequence to the device and it thus needs
    to be run before running `mx.send(stim)` (as shown in the
    function `configure_and_powerup_stim_units).

    Parameters
    ----------
    stim_unit : str
        The index of the stimulation unit to power up

    Returns
    -------
    mx.StimulationUnit
        The powered up stimulation unit

    """
    return (
        mx.StimulationUnit(stim_unit)
        .power_up(True)
        .connect(True)
        .set_voltage_mode()
        .dac_source(0)
    )


def configure_and_powerup_stim_units(stim_units: List[int]) -> List[mx.StimulationUnit]:
    """Configure and powerup the stimulation units

    Once the electrodes are connected to the stimulation units,
    the stimulation units need to be configured and powererd up.

    Parameters
    ----------
    stim_units : List[str]
        List of stimulation units indices corresponding to the connected
        stimulation electrodes

    Returns
    -------
    List[mx.StimulationUnit]
        List of stimulation unit objects corresponding to the connected
        stimulation electrodes.

    """
    stim_unit_commands: List[mx.StimulationUnit] = []
    for stim_unit in stim_units:
        stim = powerup_stim_unit(stim_unit)
        stim_unit_commands.append(stim)
        mx.send(stim)
    return stim_unit_commands


def poweroff_all_stim_units() -> None:
    """Poweroff all stimulation units

    This function is used to make sure that every stimulation units is
    powered-off before starting sequentially to send the sequences to the
    different stimulation units individually.

    Returns
    -------
    None

    """
    for stimulation_unit in range(0, 32):
        stim = mx.StimulationUnit(stimulation_unit)
        stim.power_up(False)
        stim.connect(False)
        mx.send(stim)


def create_stim_pulse(
    seq: mx.Sequence, amplitude: int, delay_samples: int, amplitude_mV: int
) -> mx.Sequence:
    """Create stimulation pulse

    The stimulation units can be controlled through three independent
    sources, what we call DAC channels (for digital analog converter).
    By programming a DAC channel with digital values, we can control
    the output the stimulation units. DAC inputs are in the range between
    0 to 1023 bits, whereas 512 corresponds to zero volt and one bit
    corresponds to 2.9 mV. When the stimulation buffers are set to voltage
    mode, they act like an inverting amplifier, hence 512 + a number of bits corresponds to a negative voltage, or 512 - a number of bits corresponds to a positive voltage.
    corresponds to a negative voltage. Thus, to give a pulse of 100mV,
    the DAC channel temporarily would need to be set to 512 - 34 (100mV/2.9)
    and back again to 512.

    Notes
    -----
    In this example, all 32 units are controlled through the same DAC
    channel ( dac_source(0) ), thus by programming a biphasic pulse
    on DAC channel 0, all the stimulation units exhibit the biphasic
    pulse.

    Parameters
    ----------
    seq : mx.Sequence
        Sequence object holding a sequence of commands, as generated
        by `mx.Sequence()`.
    amplitude : int
        Amplitude of the pulse, with units [100mV/2.9], as explained above.
    delay_samples : int
        How many samples should sand between different sequence amplitude.

    Returns
    -------
    mx.Sequence
        Sequence object filled by the pulse.

    """
    global event_counter
    event_counter += 1
    seq.append(mx.Event(0, 1, event_counter, f"amplitude {amplitude_mV} event_id {event_counter}"))
    # 先正后负还是先负后正？  反相的
    seq.append(mx.DAC(0, 512 + amplitude))
    seq.append(mx.DelaySamples(delay_samples))
    seq.append(mx.DAC(0, 512 - amplitude))
    seq.append(mx.DelaySamples(delay_samples))
    seq.append(mx.DAC(0, 512))
    return seq


def prepare_stim_sequence(
    number_pulses_per_train: int,
    inter_pulse_interval: int,
    phase: int,
    amplitude: int,
    changing_amplitude: Optional[bool] = False,
    max_amplitude: Optional[int] = None,
    amplitude_interval: Optional[int] = None,
) -> mx.Sequence:
    """Prepare a stimulation sequence.

    This is just and example and it only illustrates how sequences
    can be constructed.

    Parameters
    ----------
    number_pulses_per_train : int
        Number of repetitions of one pulse.
    inter_pulse_interval : int
        Number of samples to delay between two consecutive pulses.
    phase : int
        Number of samples before the switch from high-low (and vice versa)
        voltage when creating stimulation pulse.
    amplitude : int
        Amplitude of the pulse (minimal if changing_amplitude is True).
        Unit is millivolt.
    changing_amplitude : Optional[bool]
        Whether the pulse amplitude changes, by default False.
    max_amplitude : Optional[int]
        Maximal amplitude of the pulse if changing_amplitude is True.
        Unit is millivolt.
    amplitude_interval : Optional[int]
        Increment amplitude interval if changing_amplitude is True.
        Unit is millivolt.

    Returns
    -------
    mx.Sequence
        Sequence object filled with the stimulation sequence.

    """
    seq = mx.Sequence()
    dac_lsb_mV = float(mx.query_DAC_lsb_mV())
    if changing_amplitude:
        if max_amplitude is None or amplitude_interval is None:
            raise ValueError(
                "Both max_amplitude and amplitude_interval are required for changing_amplitude."
            )
        for cur_amplitude in range(amplitude, max_amplitude, amplitude_interval):
            for _ in range(number_pulses_per_train):
                seq = create_stim_pulse(seq, int(cur_amplitude/dac_lsb_mV), phase, cur_amplitude)
                seq.append(mx.DelaySamples(inter_pulse_interval))
            seq.append(mx.DelaySamples(inter_pulse_interval))
    else:
        for _ in range(number_pulses_per_train):
            seq = create_stim_pulse(seq, int(amplitude/dac_lsb_mV), phase, amplitude)
            seq.append(mx.DelaySamples(inter_pulse_interval))
    return seq


def send_stim_pulses_all_units(seq: mx.Sequence, number_pulse_trains: int) -> None:
    """Send stimulation pulses to all units simultaneously

    This function sends the sequence of pulses built by the
    function `prepare_stim_sequence` (the pulse train) for the
    case where the stimulation pulses are sent simultaneously to all units.

    Parameters
    ----------
    number_pulse_trains : int
        The number of repetitions of the pulse sequence.
    seq : mx.Sequence
        The sequence of pulses

    Returns
    -------
    None

    """
    for _ in range(number_pulse_trains):
        print("Send pulse")
        seq.send()
        time.sleep(10)


def send_stim_pulses_units_sequentially(
    seq: mx.Sequence, stim_units: List[int]
) -> None:
    """Send stimulation pulses to units sequentially

    This function sends the sequence of pulses built by the
    function `prepare_stim_sequence` for the case where there
    stimulation pulses are sent sequentially to one unit at a time.

    Parameters
    ----------
    seq : mx.Sequence
        The sequence of pulses
    stim_units : List[int]
        List of stimulation units indices corresponding to the connected
        stimulation electrodes

    Returns
    -------
    None

    """
    for stim_unit in stim_units:
        print(f"Power up stimulation unit {stim_unit}")
        stim = mx.StimulationUnit(stim_unit)
        stim.power_up(True).connect(True).set_voltage_mode().dac_source(0)
        mx.send(stim)
        print("Send pulse")
        seq.send()
        print(f"Power down stimulation unit {stim_unit}")
        stim = mx.StimulationUnit(stim_unit).power_up(False)
        mx.send(stim)
        time.sleep(2)


# =============================================== SystemAPI ===================================================

# ================================ random stim API ====================================
import random
def stimulate_units_random_order(
    seq: mx.Sequence,
    stim_units: List[int],
    stim_electrodes: List[int],
    repeats: int = 5,
    sleep_between_units_s: float = 10.0,
) -> List[List[int]]:
    """
    Randomly stimulate one stimulation unit at a time.
    Returns the randomized electrode order for each repeat, e.g. [[3021, 3207, 3304, 3114], ...]
    """
    # 建立 electrode -> stim_unit 的映射，方便输出的是电极编号
    el2unit = {el: unit for el, unit in zip(stim_electrodes, stim_units)}
    # stim_electrodes = [3021, 3114, 3207, 3304]
    # stim_units = [5, 8, 12, 19]
    # el2unit = {
    #     3021: 5,
    #     3114: 8,
    #     3207: 12,
    #     3304: 19
    # }

    all_orders: List[List[int]] = []

    # 保险：先把所有 unit 都断开（保持 power_up 状态不变，只切 connect）
    for u in stim_units:
        mx.send(mx.StimulationUnit(u).connect(False))

    for _ in range(repeats):
        # 生成本轮随机电极顺序，并保存
        order = stim_electrodes.copy()
        random.shuffle(order)
        all_orders.append(order)

        for el in order:
            u = el2unit[el]

            # 1) 断开所有 unit，确保只有一个接收 DAC0
            for uu in stim_units:
                mx.send(mx.StimulationUnit(uu).connect(False))

            # 2) 连接当前要刺激的 unit（它仍然是 dac_source(0)）
            mx.send(mx.StimulationUnit(u).connect(True))

            # 3) 发送一次刺激序列
            print(f"Stimulate electrode {el} (stim_unit {u})")
            seq.send()

            # 4) 等待 10s 再刺激下一个
            time.sleep(sleep_between_units_s)

    # 全部断开，防止留下连接状态
    for u in stim_units:
        mx.send(mx.StimulationUnit(u).connect(False))

    return all_orders

def setup_routing_and_units(cfg: dict) -> tuple[mx.Array, list[int]]:
    """
    负责：routing + stim_electrode->stim_unit 映射 + download + offset + clear_events
    返回：array, stim_units
    """
    wells = cfg["wells"]
    rec_electrodes = cfg["recording_electrodes"]
    stim_electrodes = cfg["stim_electrodes_32"]

    array = configure_array(rec_electrodes, stim_electrodes)

    mx.activate(wells)

    stim_units = connect_stim_units_to_stim_electrodes(stim_electrodes, array)

    array.download(wells)
    time.sleep(mx.Timing.waitAfterDownload)

    mx.offset()
    time.sleep(3)
    mx.clear_events()

    return array, stim_units


def start_recording(cfg: dict) -> mx.Saving:
    """
    负责：开始保存与录制，并返回 Saving 对象
    """
    saving_cfg = cfg["saving"]

    s = mx.Saving()
    s.open_directory(saving_cfg["dir_name"])
    s.start_file(saving_cfg["file_name"])

    s.group_define(
        0,
        saving_cfg["group_name"],
        saving_cfg["group_channels"],
    )

    s.start_recording()
    print("Start recording")
    return s


def stop_recording(s: mx.Saving) -> None:
    """
    负责：停止录制并清理 saving
    """
    print("Stop recording")
    s.stop_recording()
    time.sleep(mx.Timing.waitAfterRecording)
    s.stop_file()
    s.group_delete_all()


def build_sequence_from_cfg(cfg: dict) -> mx.Sequence:
    stim_cfg = cfg["stim"]
    return prepare_stim_sequence(
        number_pulses_per_train=stim_cfg["number_pulses_per_train"],
        inter_pulse_interval=stim_cfg["inter_pulse_interval"],
        phase=stim_cfg["phase"],
        amplitude=stim_cfg["amplitude_mV"],
    )

# ================================ random stim API ====================================

# ================================ test train API =====================================
def disconnect_all_units(stim_units_all: list[int]) -> None:
    for u in stim_units_all:
        mx.send(mx.StimulationUnit(u).connect(False))
    time.sleep(1)  # 确保都断开了

def connect_units_subset(stim_units_all: list[int], subset: list[int]) -> None:
    # 先全断开，再只连子集（避免多个模式残留连接）
    disconnect_all_units(stim_units_all)
    for u in subset:
        mx.send(mx.StimulationUnit(u).connect(True))

def build_single_pulse_sequence(cfg: dict, label: str) -> mx.Sequence:
    """
    每调用一次，就生成一个包含唯一 Event 的单脉冲 seq。
    label 用来写入模式/阶段信息，比如 "TEST_mode32"、"TRAIN_pulse3"
    """
    global event_counter

    stim_cfg = cfg["stim_pulse"]
    dac_lsb_mV = float(mx.query_DAC_lsb_mV())
    amp_bits = int(stim_cfg["amplitude_mV"] / dac_lsb_mV)
    phase = stim_cfg["phase"]
    dac_channel = stim_cfg["dac_channel"]

    seq = mx.Sequence()

    event_counter += 1
    seq.append(mx.Event(0, 1, event_counter, f"label {label} event_id {event_counter} amp_mV {stim_cfg['amplitude_mV']}"))
    # seq.append(mx.Event(0, 1, event_counter, f"amplitude {event_counter} event_id {event_counter}"))
    # biphasic pulse 
    seq.append(mx.DAC(dac_channel, 512 + amp_bits))
    seq.append(mx.DelaySamples(phase))
    seq.append(mx.DAC(dac_channel, 512 - amp_bits))
    seq.append(mx.DelaySamples(phase))
    seq.append(mx.DAC(dac_channel, 512))

    return seq

import random, time

def run_test_block(cfg: dict, stim_units_all: list[int], el2unit: dict[int, int]) -> list[list[str]]:
    """
    返回：每个 repeat 的模式顺序标签，例如:
    [
      ["mode32","mode8","mode24","mode16"],
      ...
    ]
    """
    test_cfg = cfg["test"]
    repeats = test_cfg["repeats"]
    gap_s = test_cfg["sleep_between_modes_s"]

    all_orders: list[list[str]] = []

    for r in range(repeats):
        modes = test_cfg["modes"].copy()
        random.shuffle(modes)

        order_labels = [m["name"] for m in modes]
        all_orders.append(order_labels)

        for m in modes:
            # 1) 取该模式的电极 -> unit 子集
            units_subset = [el2unit[el] for el in m["electrodes"]]

            # 2) 只连接该子集（保证“同时刺激”只作用于这些电极）
            connect_units_subset(stim_units_all, units_subset)

            # 3) 构建一次性 seq（带唯一 event + label），并 send
            seq = build_single_pulse_sequence(cfg, label=f"TEST_repeats{r+1}_{m['name']}")
            print(f"[TEST] repeats {r+1} stimulate {m['name']} electrode_nums={len(m['electrodes'])}")
            seq.send()

            # 4) 模式间隔
            time.sleep(gap_s)

    # 收尾：断开所有
    disconnect_all_units(stim_units_all)
    return all_orders

def run_train_block(cfg: dict, stim_units_all: list[int]) -> None:
    train_cfg = cfg["train"]
    bursts = train_cfg["bursts"]
    pulses_per_burst = train_cfg["pulses_per_burst"]
    freq_hz = train_cfg["freq_hz"]
    rest_s = train_cfg["rest_between_bursts_s"]
    isi_s = 1.0 / freq_hz

    # 训练始终是全32同时刺激：连接全部 unit
    connect_units_subset(stim_units_all, stim_units_all)

    for b in range(bursts):
        print(f"[TRAIN] burst {b+1}/{bursts}")

        for p in range(pulses_per_burst):
            seq = build_single_pulse_sequence(cfg, label=f"TRAIN_b{b+1}_p{p+1}")
            seq.send()
            time.sleep(isi_s)

        time.sleep(rest_s)

    disconnect_all_units(stim_units_all)

# ================================ test train API =====================================

# ================================ 所有的配置参数 =====================================
recording_electrodes = [4896,4897,4898,4899,4900,4901,4902,4903,4904,4905,4906,4907,4908,4909,4910,4911,4912,4913,4914,4915,4916,4917,4918,4919,4920,4921,4922,4923,4924,4925,4926,4927,4928,4929,4930,4931,4932,4933,4934,4935,4936,5116,5117,5118,5119,5120,5121,5122,5123,5124,5125,5126,5127,5128,5129,5130,5131,5132,5133,5134,5135,5136,5137,5138,5139,5140,5141,5142,5143,5144,5145,5146,5147,5148,5149,5150,5151,5152,5153,5154,5155,5156,5336,5337,5338,5339,5340,5341,5342,5343,5344,5345,5346,5347,5348,5349,5350,5351,5352,5353,5354,5355,5356,5357,5358,5359,5360,5361,5362,5363,5364,5365,5366,5367,5368,5369,5370,5371,5372,5373,5374,5375,5376,5556,5557,5558,5559,5560,5561,5562,5563,5564,5565,5566,5567,5568,5569,5570,5571,5572,5573,5574,5575,5576,5577,5578,5579,5580,5581,5582,5583,5584,5585,5586,5587,5588,5589,5590,5591,5592,5593,5594,5595,5596,5776,5777,5778,5779,5780,5781,5782,5783,5784,5785,5786,5787,5788,5789,5790,5791,5792,5793,5794,5795,5796,5797,5798,5799,5800,5801,5802,5803,5804,5805,5806,5807,5808,5809,5810,5811,5812,5813,5814,5815,5816,5996,5997,5998,5999,6000,6001,6002,6003,6004,6005,6006,6007,6008,6009,6010,6011,6012,6013,6014,6015,6016,6017,6018,6019,6020,6021,6022,6023,6024,6025,6026,6027,6028,6029,6030,6031,6032,6033,6034,6035,6036,6216,6217,6218,6219,6220,6221,6222,6223,6224,6225,6226,6227,6228,6229,6230,6231,6232,6233,6234,6235,6236,6237,6238,6239,6240,6241,6242,6243,6244,6245,6246,6247,6248,6249,6250,6251,6252,6253,6254,6255,6256,6436,6437,6438,6439,6440,6441,6442,6443,6444,6445,6446,6447,6448,6449,6450,6451,6452,6453,6454,6455,6456,6457,6458,6459,6460,6461,6462,6463,6464,6465,6466,6467,6468,6469,6470,6471,6472,6473,6474,6475,6476,6656,6657,6658,6659,6660,6661,6662,6663,6664,6665,6666,6667,6668,6669,6670,6671,6672,6673,6674,6675,6676,6677,6678,6679,6680,6681,6682,6683,6684,6685,6686,6687,6688,6689,6690,6691,6692,6693,6694,6695,6696,6876,6877,6878,6879,6880,6881,6882,6883,6884,6885,6886,6887,6888,6889,6890,6891,6892,6893,6894,6895,6896,6897,6898,6899,6900,6901,6902,6903,6904,6905,6906,6907,6908,6909,6910,6911,6912,6913,6914,6915,6916,7096,7097,7098,7099,7100,7101,7102,7103,7104,7105,7106,7107,7108,7109,7110,7111,7112,7113,7114,7115,7116,7117,7118,7119,7120,7121,7122,7123,7124,7125,7126,7127,7128,7129,7130,7131,7132,7133,7134,7135,7136,7316,7317,7318,7319,7320,7321,7322,7323,7324,7325,7326,7327,7328,7329,7330,7331,7332,7333,7334,7335,7336,7337,7338,7339,7340,7341,7342,7343,7344,7345,7346,7347,7348,7349,7350,7351,7352,7353,7354,7355,7356,7536,7537,7538,7539,7540,7541,7542,7543,7544,7545,7546,7547,7548,7549,7550,7551,7552,7553,7554,7555,7556,7557,7558,7559,7560,7561,7562,7563,7564,7565,7566,7567,7568,7569,7570,7571,7572,7573,7574,7575,7576,7756,7757,7758,7759,7760,7761,7762,7763,7764,7765,7766,7767,7768,7769,7770,7771,7772,7773,7774,7775,7776,7777,7778,7779,7780,7781,7782,7783,7784,7785,7786,7787,7788,7789,7790,7791,7792,7793,7794,7795,7796,7976,7977,7978,7979,7980,7981,7982,7983,7984,7985,7986,7987,7988,7989,7990,7991,7992,7993,7994,7995,7996,7997,7998,7999,8000,8001,8002,8003,8004,8005,8006,8007,8008,8009,8010,8011,8012,8013,8014,8015,8016,8196,8197,8198,8199,8200,8201,8202,8203,8204,8205,8206,8207,8208,8209,8210,8211,8212,8213,8214,8215,8216,8217,8218,8219,8220,8221,8222,8223,8224,8225,8226,8227,8228,8229,8230,8231,8232,8233,8234,8235,8236,8416,8417,8418,8419,8420,8421,8422,8423,8424,8425,8426,8427,8428,8429,8430,8431,8432,8433,8434,8435,8436,8437,8438,8439,8440,8441,8442,8443,8444,8445,8446,8447,8448,8449,8450,8451,8452,8453,8454,8455,8456,8636,8637,8638,8639,8640,8641,8642,8643,8644,8645,8646,8647,8648,8649,8650,8651,8652,8653,8654,8655,8656,8657,8658,8659,8660,8661,8662,8663,8664,8665,8666,8667,8668,8669,8670,8671,8672,8673,8674,8675,8676,8856,8857,8858,8859,8860,8861,8862,8863,8864,8865,8866,8867,8868,8869,8870,8871,8872,8873,8874,8875,8876,8877,8878,8879,8880,8881,8882,8883,8884,8885,8886,8887,8888,8889,8890,8891,8892,8893,8894,8895,8896,9076,9077,9078,9079,9080,9081,9082,9083,9084,9085,9086,9087,9088,9089,9090,9091,9092,9093,9094,9095,9096,9097,9098,9099,9100,9101,9102,9103,9104,9105,9106,9107,9108,9109,9110,9111,9112,9113,9114,9115,9116,9296,9297,9298,9299,9300,9301,9302,9303,9304,9305,9306,9307,9308,9309,9310,9311,9312,9313,9314,9315,9316,9317,9318,9319,9320,9321,9322,9323,9324,9325,9326,9327,9328,9329,9330,9331,9332,9333,9334,9335,9336,9516,9517,9518,9519,9520,9521,9522,9523,9524,9525,9526,9527,9528,9529,9530,9531,9532,9533,9534,9535,9536,9537,9538,9539,9540,9541,9542,9543,9544,9545,9546,9547,9548,9549,9550,9551,9552,9553,9554,9555,9556,9736,9737,9738,9739,9740,9741,9742,9743,9744,9745,9746,9747,9748,9749,9750,9751,9752,9753,9754,9755,9756,9757,9758,9759,9760,9761,9762,9763,9764,9765,9766,9767,9768,9769,9770,9771,9772,9773,9774,9775,9776,9956,9957,9958,9959,9960,9961,9962,9963,9964,9965,9966,9967,9968,9969,9970,9971,9972,9973,9974,9975,9976,9977,9978,9979,9980,9981,9982,9983,9984,9985,9986,9987,9988,9989,9990,9991,9992,9993,9994,9995,9996]
stim_electrodes_32 = [3158,3167,3363,3819,3828,4019,4253,4263,4273,4673,4692,4698,4709,4939,5993,6037,6873,6920,7753,7797,8633,9119,9513,10393,10400,10406,10411,10416,10421,10426,10431,10437]

mode32_list = [3158,3167,3363,3819,3828,4019,4253,4263,4273,4673,4692,4698,4709,4939,5993,6037,6873,6920,7753,7797,8633,9119,9513,10393,10400,10406,10411,10416,10421,10426,10431,10437]
mode24_list = [3158,3167,3363,3819,3828,4019,4253,4263,4273,4673,4692,4698,4709,4939,5993,6037,6873,6920,7753,7797,8633,9119,9513]
mode16_list = [3158,3167,3363,3819,3828,4019,4253,4263,4273,4673,4692,4698,4709,4939,5993,6037]
mode8_list = [3158,3167,3363,3819,3828,4019,4253,4263]

# recording_electrodes = [4896,4897,4898,4899,4900,4901,4902,4903,4904,4905,4906,4907,4908,4909,4910,4911,4912,4913,4914,4915,4916,4917,4918,4919,4920,4921,4922,4923,4924,4925,4926,4927,4928,4929,4930,4931,4932,4933,4934,4935,4936,5116,5117,5118,5119,5120,5121,5122,5123,5124,5125,5126,5127,5128,5129,5130,5131,5132,5133,5134,5135,5136,5137,5138,5139,5140,5141,5142,5143,5144,5145,5146,5147,5148,5149,5150,5151,5152,5153,5154,5155,5156,5336,5337,5338,5339,5340,5341,5342,5343,5344,5345,5346,5347,5348,5349,5350,5351,5352,5353,5354,5355,5356,5357,5358,5359,5360,5361,5362,5363,5364,5365,5366,5367,5368,5369,5370,5371,5372,5373,5374,5375,5376,5556,5557,5558,5559,5560,5561,5562,5563,5564,5565,5566,5567,5568,5569,5570,5571,5572,5573,5574,5575,5576,5577,5578,5579,5580,5581,5582,5583,5584,5585,5586,5587,5588,5589,5590,5591,5592,5593,5594,5595,5596,5776,5777,5778,5779,5780,5781,5782,5783,5784,5785,5786,5787,5788,5789,5790,5791,5792,5793,5794,5795,5796,5797,5798,5799,5800,5801,5802,5803,5804,5805,5806,5807,5808,5809,5810,5811,5812,5813,5814,5815,5816,5996,5997,5998,5999,6000,6001,6002,6003,6004,6005,6006,6007,6008,6009,6010,6011,6012,6013,6014,6015,6016,6017,6018,6019,6020,6021,6022,6023,6024,6025,6026,6027,6028,6029,6030,6031,6032,6033,6034,6035,6036,6216,6217,6218,6219,6220,6221,6222,6223,6224,6225,6226,6227,6228,6229,6230,6231,6232,6233,6234,6235,6236,6237,6238,6239,6240,6241,6242,6243,6244,6245,6246,6247,6248,6249,6250,6251,6252,6253,6254,6255,6256,6436,6437,6438,6439,6440,6441,6442,6443,6444,6445,6446,6447,6448,6449,6450,6451,6452,6453,6454,6455,6456,6457,6458,6459,6460,6461,6462,6463,6464,6465,6466,6467,6468,6469,6470,6471,6472,6473,6474,6475,6476,6656,6657,6658,6659,6660,6661,6662,6663,6664,6665,6666,6667,6668,6669,6670,6671,6672,6673,6674,6675,6676,6677,6678,6679,6680,6681,6682,6683,6684,6685,6686,6687,6688,6689,6690,6691,6692,6693,6694,6695,6696,6876,6877,6878,6879,6880,6881,6882,6883,6884,6885,6886,6887,6888,6889,6890,6891,6892,6893,6894,6895,6896,6897,6898,6899,6900,6901,6902,6903,6904,6905,6906,6907,6908,6909,6910,6911,6912,6913,6914,6915,6916,7096,7097,7098,7099,7100,7101,7102,7103,7104,7105,7106,7107,7108,7109,7110,7111,7112,7113,7114,7115,7116,7117,7118,7119,7120,7121,7122,7123,7124,7125,7126,7127,7128,7129,7130,7131,7132,7133,7134,7135,7136,7316,7317,7318,7319,7320,7321,7322,7323,7324,7325,7326,7327,7328,7329,7330,7331,7332,7333,7334,7335,7336,7337,7338,7339,7340,7341,7342,7343,7344,7345,7346,7347,7348,7349,7350,7351,7352,7353,7354,7355,7356,7536,7537,7538,7539,7540,7541,7542,7543,7544,7545,7546,7547,7548,7549,7550,7551,7552,7553,7554,7555,7556,7557,7558,7559,7560,7561,7562,7563,7564,7565,7566,7567,7568,7569,7570,7571,7572,7573,7574,7575,7576,7756,7757,7758,7759,7760,7761,7762,7763,7764,7765,7766,7767,7768,7769,7770,7771,7772,7773,7774,7775,7776,7777,7778,7779,7780,7781,7782,7783,7784,7785,7786,7787,7788,7789,7790,7791,7792,7793,7794,7795,7796,7976,7977,7978,7979,7980,7981,7982,7983,7984,7985,7986,7987,7988,7989,7990,7991,7992,7993,7994,7995,7996,7997,7998,7999,8000,8001,8002,8003,8004,8005,8006,8007,8008,8009,8010,8011,8012,8013,8014,8015,8016,8196,8197,8198,8199,8200,8201,8202,8203,8204,8205,8206,8207,8208,8209,8210,8211,8212,8213,8214,8215,8216,8217,8218,8219,8220,8221,8222,8223,8224,8225,8226,8227,8228,8229,8230,8231,8232,8233,8234,8235,8236,8416,8417,8418,8419,8420,8421,8422,8423,8424,8425,8426,8427,8428,8429,8430,8431,8432,8433,8434,8435,8436,8437,8438,8439,8440,8441,8442,8443,8444,8445,8446,8447,8448,8449,8450,8451,8452,8453,8454,8455,8456,8636,8637,8638,8639,8640,8641,8642,8643,8644,8645,8646,8647,8648,8649,8650,8651,8652,8653,8654,8655,8656,8657,8658,8659,8660,8661,8662,8663,8664,8665,8666,8667,8668,8669,8670,8671,8672,8673,8674,8675,8676,8856,8857,8858,8859,8860,8861,8862,8863,8864,8865,8866,8867,8868,8869,8870,8871,8872,8873,8874,8875,8876,8877,8878,8879,8880,8881,8882,8883,8884,8885,8886,8887,8888,8889,8890,8891,8892,8893,8894,8895,8896,9076,9077,9078,9079,9080,9081,9082,9083,9084,9085,9086,9087,9088,9089,9090,9091,9092,9093,9094,9095,9096,9097,9098,9099,9100,9101,9102,9103,9104,9105,9106,9107,9108,9109,9110,9111,9112,9113,9114,9115,9116,9296,9297,9298,9299,9300,9301,9302,9303,9304,9305,9306,9307,9308,9309,9310,9311,9312,9313,9314,9315,9316,9317,9318,9319,9320,9321,9322,9323,9324,9325,9326,9327,9328,9329,9330,9331,9332,9333,9334,9335,9336,9516,9517,9518,9519,9520,9521,9522,9523,9524,9525,9526,9527,9528,9529,9530,9531,9532,9533,9534,9535,9536,9537,9538,9539,9540,9541,9542,9543,9544,9545,9546,9547,9548,9549,9550,9551,9552,9553,9554,9555,9556,9736,9737,9738,9739,9740,9741,9742,9743,9744,9745,9746,9747,9748,9749,9750,9751,9752,9753,9754,9755,9756,9757,9758,9759,9760,9761,9762,9763,9764,9765,9766,9767,9768,9769,9770,9771,9772,9773,9774,9775,9776,9956,9957,9958,9959,9960,9961,9962,9963,9964,9965,9966,9967,9968,9969,9970,9971,9972,9973,9974,9975,9976,9977,9978,9979,9980,9981,9982,9983,9984,9985,9986,9987,9988,9989,9990,9991,9992,9993,9994,9995,9996]
# stim_electrodes_32 = [4895, 4938, 9997, 9413]

# mode32_list = [4895, 4938, 9997, 9413]
# mode24_list = [4895, 4938, 9997]
# mode16_list = [4895, 4938]
# mode8_list = [4895]

event_counter = 1 # 刺激时刻事件计数器，每来一个刺激事件就加1，从1开始 不用改
CONFIG = {
    "wells": [0],
    "recording_electrodes": recording_electrodes,
    "stim_electrodes_32": stim_electrodes_32,

    "saving": {
        "dir_name": "/home/maxwell/Data/test_recording/",
        "file_name": "exp_test_train_mix",
        "group_name": "all_channels",
        "group_channels": list(range(1024)),
    },
    # 随机刺激参数
    "stim": {
        "number_pulses_per_train": 1,
        "inter_pulse_interval": 0,
        "phase": 4, # 4 samples = 200us
        "amplitude_mV": 500, # mV
        "repeats": 5, # 整个随机过程的重复次数
        "sleep_between_units_s": 10.0, # 10s
        "dac_channel": 0,
    },

    # 单脉冲参数（test/train 都复用） 应该是脉冲参数都一样？然后频率刺激空间不一样？
    "stim_pulse": {
        "phase": 4, # 4 samples = 200us
        "amplitude_mV": 500, # mV
        "dac_channel": 0,
    },

    "test": {
        "repeats": 5, 
        "sleep_between_modes_s": 10.0,
        "modes": [
            {"name": "mode32", "electrodes": mode32_list},
            {"name": "mode24", "electrodes": mode24_list},
            {"name": "mode16", "electrodes": mode16_list},
            {"name": "mode8",  "electrodes": mode8_list},
        ],
    },

    "train": {
        "freq_hz": 1.0, # 1hz 刺激
        "pulses_per_burst": 10, # 每个 burst 里刺激 10 个脉冲，1hz 脉冲刺激10次
        "rest_between_bursts_s": 10.0, # 每个 burst 之间休息 10s
        "bursts": 10, # 共重复10次 
    },

    "protocol": {
        "cycles": 7,                 # (train + rest + test) 重复次数
        "rest_after_test_s": 300.0,  # 秒 test 后休息 5min
        "rest_after_train_s": 300.0, # 秒 train 后休息 5min
    },
}

# ================================ 所有的配置参数 =====================================

# 1 随机刺激实验流程控制
def run_random_stim_experiment(cfg: dict) -> list[list[int]]:
    """
    负责：完整跑一次实验（初始化系统 -> routing -> recording -> random stim -> stop）
    返回：allorder
    """
    initialize_system()

    _, stim_units = setup_routing_and_units(cfg)

    s = start_recording(cfg)

    # 构建刺激序列
    seq = build_sequence_from_cfg(cfg)

    # 刺激单元 & dac配置 powerup + dac_source
    configure_and_powerup_stim_units(stim_units)
    # 加载配置参数并随机刺激
    stim_cfg = cfg["stim"]
    allorder = stimulate_units_random_order(
        seq=seq,
        stim_units=stim_units,
        stim_electrodes=cfg["stim_electrodes_32"],
        repeats=stim_cfg["repeats"],
        sleep_between_units_s=stim_cfg["sleep_between_units_s"],
    )

    stop_recording(s)

    return allorder

# 2 缺失刺激实验流程控制（test + train）
def run_protocol(cfg: dict) -> dict:
    """
    固定实验流程：
      test + rest_5min +  (train +  rest_5min +  test) * 7

    返回一个 dict，后面再加保存逻辑，先print出来手动保存一下：
      {
        "test_orders": [ ... 每次 test 的模式顺序 ... ],
        "protocol_steps": [ ... 记录执行了哪些步骤 ... ]
      }
    """
    proto_cfg = cfg["protocol"]

    # ====== 0) 系统初始化 ======
    initialize_system()

    # ====== 1) routing + stim_units ======
    _, stim_units = setup_routing_and_units(cfg)

    # ====== 2) 开始录制 ======
    s = start_recording(cfg)

    # ====== 3) 上电/配置 stim units（dac_source 等）======
    configure_and_powerup_stim_units(stim_units)

    # ====== 4) 建 el->unit 映射（用于 test 的模式子集选择）======
    stim_electrodes_32 = cfg["stim_electrodes_32"]
    el2unit = {el: u for el, u in zip(stim_electrodes_32, stim_units)}

    # ====== 5) 执行固定流程 ======
    results = {
        "test_orders": [],       # 每次 test 的 mode 顺序标签（test_block 返回的那个）
        "protocol_steps": [],    # 纯日志：记录跑了哪些步骤
    }

    def do_rest(seconds: float, tag: str):
        print(f"[REST] {tag}: {seconds:.1f}s")
        results["protocol_steps"].append({"type": "rest", "tag": tag, "seconds": seconds})
        time.sleep(seconds)

    # ---- Step A: test ----
    print("[FLOW] TEST #0")
    results["protocol_steps"].append({"type": "test", "index": 0})
    test_orders = run_test_block(cfg, stim_units_all=stim_units, el2unit=el2unit)
    results["test_orders"].append({
        "test_index": 1,          # 第1次 test
        "mode_order": test_orders,  # 这一轮 repeat 内部每次的 mode 顺序
    })

    # ---- Step B: rest 5min ----
    do_rest(proto_cfg["rest_after_test_s"], tag="after_test0")

    # ---- Step C: (train + rest + test) * 7 ----
    n_cycles = proto_cfg["cycles"]
    for i in range(1, n_cycles + 1):
        print(f"[FLOW] TRAIN #{i}")
        results["protocol_steps"].append({"type": "train", "index": i})
        run_train_block(cfg, stim_units_all=stim_units)

        do_rest(proto_cfg["rest_after_train_s"], tag=f"after_train{i}")

        print(f"[FLOW] TEST #{i}")
        results["protocol_steps"].append({"type": "test", "index": i})
        test_orders = run_test_block(cfg, stim_units_all=stim_units, el2unit=el2unit)
        results["test_orders"].append({
            "test_index": i+1,          # 第i次 test
            "mode_order": test_orders,  # 这一轮 repeat 内部每次的 mode 顺序
        })

    # ====== 6) 停止录制 ======
    stop_recording(s)

    return results


def main():
    # 1 随机刺激实验
    # allorder = run_random_stim_experiment(CONFIG)
    # print("Random orders:", allorder)

    # 2 test train 实验 进行之前前两个注意需要注释掉
    results = run_protocol(CONFIG)
    print("Protocol results:", results)


if __name__ == "__main__":
    main()

