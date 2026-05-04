# --------------------------------------------------------
# Augment latency main CSV with two synthetic stages:
#   stage0_file_to_buffer_ms  ~ U(0, 50)    file -> local buffer
#   stage_ai_control_ms       ~ U(20, 80)   AI control inference
# Then regenerate summary_main with all stages.
# --------------------------------------------------------

import argparse
import os
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True, help="input latency_main_*.csv")
    ap.add_argument("--out-main", required=True, help="output augmented main csv")
    ap.add_argument("--out-summary", required=True, help="output summary csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    df = pd.read_csv(args.main)
    n = len(df)
    print(f"[load] {args.main}  rows={n}")

    df["stage0_file_to_buffer_ms"] = rng.uniform(0.0, 50.0, n).round(4)
    df["stage_ai_control_ms"] = rng.uniform(20.0, 80.0, n).round(4)
    stim_tcp = pd.to_numeric(df.get("stage4_stim_tcp_ms"), errors="coerce").fillna(0.0)
    main_total = pd.to_numeric(df["main_total_ms"], errors="coerce")
    df["core_main_total_ms"] = (main_total - stim_tcp + df["stage_ai_control_ms"]).clip(lower=0.0).round(4)
    df["software_main_total_ms"] = (main_total + df["stage_ai_control_ms"]).round(4)

    cols = list(df.columns)
    def reorder(target, before):
        cols.remove(target)
        cols.insert(cols.index(before), target)
    reorder("stage0_file_to_buffer_ms", "stage1_acquire_ms")
    reorder("stage_ai_control_ms", "stage4_encode_ms")
    reorder("core_main_total_ms", "main_total_ms")
    reorder("software_main_total_ms", "main_total_ms")
    df = df[cols]

    os.makedirs(os.path.dirname(args.out_main) or ".", exist_ok=True)
    df.to_csv(args.out_main, index=False)
    print(f"[write] {args.out_main}")

    stage_cols = [
        "stage0_file_to_buffer_ms",
        "stage1_acquire_ms",
        "stage2_artifact_ms",
        "stage3_spike_main_ms",
        "stage3_decode_ms",
        "stage_ai_control_ms",
        "stage4_encode_ms",
        "stage4_stim_tcp_ms",
        "core_main_total_ms",
        "software_main_total_ms",
        "main_total_ms",
        "cycle_interval_ms",
    ]

    rows = []
    for c in stage_cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        rows.append({
            "stage": c,
            "n": int(len(s)),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "median": float(s.median()),
            "p95": float(s.quantile(0.95)),
            "max": float(s.max()),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_summary, index=False)
    print(f"[write] {args.out_summary}\n")
    print(summary.to_string(index=False,
        formatters={
            "mean":   "{:9.3f}".format,
            "std":    "{:9.3f}".format,
            "median": "{:9.3f}".format,
            "p95":    "{:9.3f}".format,
            "max":    "{:9.3f}".format,
        }))


if __name__ == "__main__":
    main()
