# --------------------------------------------------------
# Offline latency analyzer for thesis section 4.3.2.
# Works on partial CSVs, no minimum length required.
# --------------------------------------------------------
"""
Usage:
    python scripts/analyze_latency.py \
        --main out_latency/latency_main_<ts>.csv \
        --spike out_latency/latency_spike_async_<ts>.csv \
        --out out_latency/figs_<ts>

If --spike is omitted, the async spike report is skipped.
If --out is omitted, figures are written next to the main CSV.
"""

import argparse
import os
import sys

try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError as exc:
    print(f"[ERR] missing Python dependency: {exc.name}", file=sys.stderr)
    print("Install/use an environment with numpy and pandas before running this analyzer.", file=sys.stderr)
    sys.exit(1)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STAGE_DEFS = [
    ("stage0_file_to_buffer_ms", "0. file->buffer"),
    ("stage1_acquire_ms", "1. acquire"),
    ("stage2_artifact_ms", "2. artifact"),
    ("stage3_spike_main_ms", "3a. spike (main)"),
    ("stage3_decode_ms", "3b. decode"),
    ("stage_ai_control_ms", "3c. AI control"),
    ("stage4_encode_ms", "4a. encode"),
    ("stage4_stim_tcp_ms", "4b. stim TCP"),
]
STAGE_COLS = [col for col, _ in STAGE_DEFS]

CONTROL_PERIOD_MS = 100.0


def enrich_latency_columns(df):
    df = df.copy()
    main_total = pd.to_numeric(df.get("main_total_ms"), errors="coerce")
    stim_tcp = pd.to_numeric(df.get("stage4_stim_tcp_ms"), errors="coerce")
    if "stage_ai_control_ms" in df.columns:
        ai_control = pd.to_numeric(df["stage_ai_control_ms"], errors="coerce").fillna(0.0)
    else:
        ai_control = 0.0
    stim_filled = stim_tcp.fillna(0.0)
    df["core_main_total_ms"] = (main_total - stim_filled + ai_control).clip(lower=0.0)
    df["software_main_total_ms"] = main_total + ai_control
    if "cycle_interval_ms" in df.columns:
        cycle_interval = pd.to_numeric(df["cycle_interval_ms"], errors="coerce")
        df["cycle_jitter_ms"] = cycle_interval - CONTROL_PERIOD_MS
        df["cycle_abs_jitter_ms"] = df["cycle_jitter_ms"].abs()
    df["stim_tcp_present"] = stim_tcp.notna()
    return df


def stats_block(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return None
    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        "median": float(s.median()),
        "p95": float(s.quantile(0.95)),
        "max": float(s.max()),
    }


def stats_table(df, cols):
    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        st = stats_block(df[col])
        if st is None:
            continue
        rows.append({"stage": col, **st})
    return pd.DataFrame(rows)


def plot_stage_box(df, out_path):
    data = []
    labels = []
    stage_labels = dict(STAGE_DEFS)
    core_cols = [col for col in STAGE_COLS if col != "stage4_stim_tcp_ms"]
    for col in core_cols:
        if col not in df.columns:
            continue
        lab = stage_labels[col]
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        data.append(s.values)
        labels.append(f"{lab}\n(n={len(s)})")
    if not data:
        return
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4.6),
        gridspec_kw={"width_ratios": [4.5, 1.3]}
    )

    axes[0].boxplot(data, labels=labels, showfliers=False)
    axes[0].set_ylabel("Duration (ms)")
    axes[0].set_title("Core main-loop stages")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].tick_params(axis="x", labelrotation=20)

    stim = pd.to_numeric(df.get("stage4_stim_tcp_ms"), errors="coerce").dropna()
    if not stim.empty:
        axes[1].boxplot([stim.values], labels=[f"4b. stim TCP\n(n={len(stim)})"], showfliers=True)
        axes[1].set_title("Stim command")
        axes[1].grid(True, axis="y", alpha=0.3)
    else:
        axes[1].axis("off")

    fig.suptitle("Latency distribution by processing stage", y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_total_timeline(df, out_path):
    x = pd.to_numeric(df["cycle_id"], errors="coerce")
    core = pd.to_numeric(df["core_main_total_ms"], errors="coerce")
    full = pd.to_numeric(df["software_main_total_ms"], errors="coerce")
    stim = pd.to_numeric(df.get("stage4_stim_tcp_ms"), errors="coerce")
    stim_mask = stim.notna()

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(x, core, lw=0.8, color="#2F6B9A", label="core total (stim TCP excluded)")
    axes[0].axhline(CONTROL_PERIOD_MS, color="r", ls="--", lw=1, label=f"{CONTROL_PERIOD_MS:.0f} ms control period")
    axes[0].set_ylabel("Core duration (ms)")
    axes[0].set_title("Core closed-loop latency per cycle")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    if stim_mask.any():
        axes[1].scatter(x[stim_mask], stim[stim_mask], s=10, color="#B75D3B", alpha=0.75, label="stim TCP command")
        axes[1].scatter(x[stim_mask], full[stim_mask], s=8, color="#777777", alpha=0.35, label="software total on stim cycles")
    axes[1].axhline(CONTROL_PERIOD_MS, color="r", ls="--", lw=1, label=f"{CONTROL_PERIOD_MS:.0f} ms reference")
    axes[1].set_xlabel("Control cycle index")
    axes[1].set_ylabel("Blocking duration (ms)")
    axes[1].set_title("Stimulus-command blocking cycles")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_cycle_interval(df, out_path):
    s = pd.to_numeric(df["cycle_interval_ms"], errors="coerce")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df["cycle_id"], s, lw=0.7)
    ax.axhline(CONTROL_PERIOD_MS, color="r", ls="--", lw=1, label=f"Nominal {CONTROL_PERIOD_MS:.0f} ms")
    ax.set_xlabel("Control cycle index")
    ax.set_ylabel("Actual cycle interval (ms)")
    ax.set_title("QTimer cycle interval (jitter probe)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_spike_async(df, out_path):
    s = pd.to_numeric(df["run_duration_ms"], errors="coerce").dropna()
    if s.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(s.values, bins=50)
    axes[0].set_xlabel("Single async spike-detection duration (ms)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Distribution (n={len(s)})")
    axes[0].grid(True, alpha=0.3)

    x = np.arange(len(s))
    normal = s <= CONTROL_PERIOD_MS
    axes[1].scatter(x[normal], s[normal], s=7, alpha=0.7, label="<= 100 ms")
    if (~normal).any():
        axes[1].scatter(x[~normal], s[~normal], s=14, color="#B75D3B", label="> 100 ms")
    axes[1].axhline(CONTROL_PERIOD_MS, color="r", ls="--", lw=1, label=f"{CONTROL_PERIOD_MS:.0f} ms reference")
    axes[1].set_xlabel("Spike-run index")
    axes[1].set_ylabel("Duration (ms)")
    axes[1].set_title("Spike-run timeline")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True, help="Path to latency_main_<ts>.csv")
    ap.add_argument("--spike", default=None, help="Path to latency_spike_async_<ts>.csv")
    ap.add_argument("--out", default=None, help="Output dir for figures and stats")
    args = ap.parse_args()

    if not os.path.isfile(args.main):
        print(f"[ERR] main CSV not found: {args.main}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(args.main)), "figs")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(args.main)
    df = enrich_latency_columns(df)
    print(f"[main] cycles loaded: {len(df)}")
    if df.empty:
        print("[main] empty CSV; nothing to do.")
        return

    summary_main = stats_table(
        df,
        STAGE_COLS + [
            "protocol_wait_ms",
            "core_main_total_ms",
            "software_main_total_ms",
            "main_total_ms",
            "main_total_raw_ms",
            "cycle_interval_ms",
            "cycle_jitter_ms",
            "cycle_abs_jitter_ms",
        ],
    )
    print("\n=== Main-loop per-stage statistics (ms) ===")
    if not summary_main.empty:
        with pd.option_context("display.float_format", "{:8.3f}".format):
            print(summary_main.to_string(index=False))
    summary_main.to_csv(os.path.join(out_dir, "summary_main.csv"), index=False)

    over = pd.to_numeric(df["software_main_total_ms"], errors="coerce") > CONTROL_PERIOD_MS
    over_count = int(over.sum())
    print(f"\nSoftware-total cycles exceeding {CONTROL_PERIOD_MS:.0f} ms: {over_count} / {len(df)} "
          f"({100.0 * over_count / max(len(df), 1):.2f}%)")

    core_over = pd.to_numeric(df["core_main_total_ms"], errors="coerce") > CONTROL_PERIOD_MS
    core_over_count = int(core_over.sum())
    stim_cycles = int(df["stim_tcp_present"].sum())
    print(f"Core cycles exceeding {CONTROL_PERIOD_MS:.0f} ms after excluding stim TCP: "
          f"{core_over_count} / {len(df)} ({100.0 * core_over_count / max(len(df), 1):.2f}%)")
    print(f"Cycles with stimulus TCP command: {stim_cycles} / {len(df)} "
          f"({100.0 * stim_cycles / max(len(df), 1):.2f}%)")

    plot_stage_box(df, os.path.join(out_dir, "stage_boxplot.png"))
    plot_total_timeline(df, os.path.join(out_dir, "total_timeline.png"))
    plot_cycle_interval(df, os.path.join(out_dir, "cycle_interval.png"))

    if args.spike and os.path.isfile(args.spike):
        sp = pd.read_csv(args.spike)
        print(f"\n[spike] async runs loaded: {len(sp)}")
        if not sp.empty:
            sp_stat = stats_block(sp["run_duration_ms"])
            print("\n=== Async spike-detection single-run stats (ms) ===")
            for k, v in sp_stat.items():
                print(f"  {k:>6} = {v:.3f}" if isinstance(v, float) else f"  {k:>6} = {v}")
            pd.DataFrame([sp_stat]).to_csv(os.path.join(out_dir, "summary_spike_async.csv"), index=False)
            plot_spike_async(sp, os.path.join(out_dir, "spike_async.png"))

    xlsx_path = os.path.join(out_dir, "latency_summary.xlsx")
    try:
        with pd.ExcelWriter(xlsx_path) as writer:
            summary_main.to_excel(writer, sheet_name="main_summary", index=False)
            df.to_excel(writer, sheet_name="main_cycles", index=False)
            if args.spike and os.path.isfile(args.spike):
                sp.to_excel(writer, sheet_name="spike_async", index=False)
        print(f"[write] {xlsx_path}")
    except Exception as exc:
        print(f"[warn] Excel export skipped: {exc}")

    print(f"\nFigures and summaries written to: {out_dir}")


if __name__ == "__main__":
    main()
