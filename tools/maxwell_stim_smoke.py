"""
Maxwell 真实刺激发放冒烟测试。

独立于 maxwell_phase_b_smoke.py。后者只验 MetaBOC 平台路径骨架（stub
print），本脚本验真实 mx.send DAC 序列能输出双相脉冲到指定 stim 电极。

参考：Maxwell 官方文档 examples/python/stimulate.html
  - create_stim_pulse (line 633-681)：双相脉冲构造模板
  - powerup_stim_unit (line 578-584)：StimulationUnit 状态机
  - DAC 数学：512 = 零点，1 LSB ≈ 2.9 mV (mx.query_DAC_lsb_mV() 动态查)
  - phase=4 samples = 200us (50us/sample @ 20kHz)
  - 发放：seq.send()（Sequence 对象的方法，不是 mx.send(seq)）
  - 发放前必须 mx.clear_events() (line 830)

使用
----
Linux 真机（mxwserver 运行中，建议同时开 MaxLab Live 录制方便事后查 Event）：

    python tools/maxwell_stim_smoke.py /path/to/your.cfg \\
        --stim "14589,5704" --target 14589

    可选参数：
      --amplitude-mv 200       每相幅度 (mV)
      --phase-samples 4        每相持续 (samples, 4 = 200us)
      --n-pulses 5             脉冲数
      --inter-pulse-samples 2000   间隔 (samples, 2000 = 100ms)

成功标志
--------
  - 终端依次打印 [1]→[6]，最后 RESULT: PASS
  - mx.send 全程无 'Error'
  - 若同时跑 MaxLab Live 录制，事后回看 frame 元数据可见 Event 标记
    "stim_smoke pulse_N amp_NmV"
"""

import argparse
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Maxwell stim pulse smoke.")
    parser.add_argument("cfg_path", help="MaxLab Live 导出的 .cfg 文件路径")
    parser.add_argument("--stim", required=True,
                        help="候选 stim 电极池 (comma-separated, e.g. '14589,5704')")
    parser.add_argument("--target", type=int, required=True,
                        help="发放脉冲的目标电极 (必须在 --stim 列表里)")
    parser.add_argument("--amplitude-mv", type=int, default=200,
                        help="脉冲每相幅度 (mV), 默认 200")
    parser.add_argument("--phase-samples", type=int, default=4,
                        help="每相持续 (samples), 默认 4 = 200us @ 20kHz")
    parser.add_argument("--n-pulses", type=int, default=5,
                        help="脉冲数, 默认 5")
    parser.add_argument("--inter-pulse-samples", type=int, default=2000,
                        help="间隔 (samples), 默认 2000 = 100ms @ 20kHz")
    args = parser.parse_args()

    cfg_path = Path(args.cfg_path)
    if not cfg_path.exists():
        print("cfg not found: {}".format(cfg_path))
        sys.exit(2)

    stim_electrodes = [int(x.strip()) for x in args.stim.split(",") if x.strip()]
    if args.target not in stim_electrodes:
        print("--target {} must be in --stim {}".format(args.target, stim_electrodes))
        sys.exit(2)

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    print("cfg_path        = {}".format(cfg_path))
    print("stim pool       = {}".format(stim_electrodes))
    print("target electrode= {}".format(args.target))
    print("amplitude       = {} mV / phase".format(args.amplitude_mv))
    print("phase           = {} samples ({} us)".format(
        args.phase_samples, args.phase_samples * 50))
    print("n_pulses        = {}".format(args.n_pulses))
    print("inter_pulse     = {} samples ({} ms)".format(
        args.inter_pulse_samples, args.inter_pulse_samples * 50 / 1000))

    from src.system_device.maxwell_sys import MaxwellSystem
    import maxlab as mx

    sys_obj = MaxwellSystem()
    print()
    print("=" * 60)
    print("[1] start_session — 完整启动序列 (cfg parse / route / download / power_up)")
    print("=" * 60)
    sys_obj.start_session(
        cfg_path=str(cfg_path),
        record_electrodes=None,
        stim_electrodes=stim_electrodes,
        role_mapping=None,
    )

    failed = False
    try:
        print()
        print("[2] resolve electrode {} → stim unit".format(args.target))
        unit_id = sys_obj.stim_pool.get_unit(args.target)
        print("    electrode {} → stim_unit {}".format(args.target, unit_id))

        print()
        print("[3] activate (connect_electrode_to_stimulation runtime)")
        sys_obj.stim_pool.activate([args.target])
        print("    active electrodes: {}".format(sorted(sys_obj.stim_pool.get_active_electrodes())))

        print()
        print("[4] build biphasic sequence")
        dac_lsb_mV = float(mx.query_DAC_lsb_mV())
        amp_dac = int(args.amplitude_mv / dac_lsb_mV)
        print("    DAC LSB = {:.4f} mV/bit".format(dac_lsb_mV))
        print("    amplitude in DAC bits = {} (= {} mV)".format(
            amp_dac, amp_dac * dac_lsb_mV))

        mx.clear_events()  # 与 stimulate.html line 830 标准模板对齐
        seq = mx.Sequence()
        for i in range(args.n_pulses):
            label = "stim_smoke pulse_{} amp_{}mV unit_{}".format(
                i + 1, args.amplitude_mv, unit_id)
            seq.append(mx.Event(0, 1, i + 1, label))
            # 双相脉冲：负相先（电压模式下逆相放大器，文档 line 654-655）
            seq.append(mx.DAC(0, 512 - amp_dac))
            seq.append(mx.DelaySamples(args.phase_samples))
            seq.append(mx.DAC(0, 512 + amp_dac))
            seq.append(mx.DelaySamples(args.phase_samples))
            seq.append(mx.DAC(0, 512))
            seq.append(mx.DelaySamples(args.inter_pulse_samples))
        print("    sequence built: {} biphasic pulses".format(args.n_pulses))

        print()
        print("[5] send sequence — pulses going out NOW")
        seq.send()
        # 等待全部脉冲发完再清理 (n_pulses * (2*phase + inter_pulse) samples @ 20kHz)
        total_samples = args.n_pulses * (2 * args.phase_samples + args.inter_pulse_samples)
        wait_s = total_samples * 50e-6 + 0.5  # 50us/sample + 余量
        time.sleep(wait_s)
        print("    sent. waited {:.2f} s for completion.".format(wait_s))

        print()
        print("[6] deactivate")
        sys_obj.stim_pool.deactivate([args.target])
        print("    deactivated electrode {}".format(args.target))

    except Exception:
        failed = True
        import traceback
        print()
        print("STIM SMOKE FAILED — traceback below:")
        traceback.print_exc()

    finally:
        try:
            sys_obj.stop_connect()
            print()
            print("[cleanup] stop_connect OK.")
        except Exception as exc:
            print("[cleanup] stop_connect raised: {!r}".format(exc))

    print()
    if failed:
        print("RESULT: FAIL")
        sys.exit(1)
    print("RESULT: PASS")
    print()
    print("如果 MaxLab Live 同时在录制，回看时应能看到 {} 个 Event 标记".format(args.n_pulses))
    print("(label: 'stim_smoke pulse_N ...') 与对应电极上的双相波形。")


if __name__ == "__main__":
    main()
