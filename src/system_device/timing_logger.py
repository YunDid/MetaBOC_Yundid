# --------------------------------------------------------
# Closed-loop latency logger for thesis section 4.3.2
# Streaming append CSVs, safe to interrupt at any moment.
# --------------------------------------------------------

import csv
import os
import time
import threading
from datetime import datetime


class TimingLogger:
    """
    Singleton latency logger.

    Usage in main loop:
        TimingLogger.get().begin_cycle()
        ... mark stages along the pipeline ...
        TimingLogger.get().end_cycle()

    Usage from async spike thread:
        TimingLogger.get().log_spike_async(duration_ms)

    Two CSVs are produced per process launch:
        out_latency/latency_main_<ts>.csv     - one row per main loop cycle
        out_latency/latency_spike_async_<ts>.csv - one row per async spike run

    Each row is flushed immediately so partial runs are recoverable.
    """

    _instance = None
    _instance_lock = threading.Lock()

    @classmethod
    def get(cls):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(os.getcwd(), "out_latency")
        os.makedirs(out_dir, exist_ok=True)

        self.main_csv_path = os.path.join(out_dir, f"latency_main_{ts}.csv")
        self.spike_csv_path = os.path.join(out_dir, f"latency_spike_async_{ts}.csv")

        self._main_file = open(self.main_csv_path, "w", newline="", buffering=1)
        self._main_writer = csv.writer(self._main_file)
        self._main_writer.writerow([
            "cycle_id",
            "t_wall_start",
            "stage1_acquire_ms",
            "stage2_artifact_ms",
            "stage3_spike_main_ms",
            "stage3_decode_ms",
            "stage4_encode_ms",
            "stage4_stim_tcp_ms",
            "protocol_wait_ms",
            "main_total_ms",
            "main_total_raw_ms",
            "cycle_interval_ms",
        ])

        self._spike_file = open(self.spike_csv_path, "w", newline="", buffering=1)
        self._spike_writer = csv.writer(self._spike_file)
        self._spike_writer.writerow([
            "t_wall",
            "cycle_id",
            "run_duration_ms",
        ])

        self._cycle_id = 0
        self._cur = None
        self._last_cycle_start = None
        self._spike_lock = threading.Lock()

    # ---------- main loop ----------

    def begin_cycle(self):
        now = time.perf_counter()
        interval = (now - self._last_cycle_start) * 1000.0 if self._last_cycle_start else float("nan")
        self._last_cycle_start = now
        self._cycle_id += 1
        self._cur = {
            "cycle_id": self._cycle_id,
            "t_wall_start": time.time(),
            "cycle_interval_ms": interval,
            "protocol_wait_ms": 0.0,
            "_t0": now,
        }
        return self._cycle_id

    def mark(self, stage_key, duration_ms):
        if self._cur is None:
            return
        self._cur[stage_key] = duration_ms

    def add_protocol_wait(self, duration_ms):
        if self._cur is None:
            return
        self._cur["protocol_wait_ms"] = self._cur.get("protocol_wait_ms", 0.0) + duration_ms

    def end_cycle(self):
        if self._cur is None:
            return
        raw_total = (time.perf_counter() - self._cur["_t0"]) * 1000.0
        protocol_wait = self._cur.get("protocol_wait_ms", 0.0)
        total = max(0.0, raw_total - protocol_wait)
        row = [
            self._cur.get("cycle_id"),
            f"{self._cur.get('t_wall_start', 0):.6f}",
            _fmt(self._cur.get("stage1_acquire_ms")),
            _fmt(self._cur.get("stage2_artifact_ms")),
            _fmt(self._cur.get("stage3_spike_main_ms")),
            _fmt(self._cur.get("stage3_decode_ms")),
            _fmt(self._cur.get("stage4_encode_ms")),
            _fmt(self._cur.get("stage4_stim_tcp_ms")),
            _fmt(protocol_wait),
            f"{total:.4f}",
            f"{raw_total:.4f}",
            _fmt(self._cur.get("cycle_interval_ms")),
        ]
        try:
            self._main_writer.writerow(row)
            self._main_file.flush()
        except Exception:
            pass
        self._cur = None

    # ---------- async spike thread ----------

    def log_spike_async(self, run_duration_ms):
        with self._spike_lock:
            try:
                self._spike_writer.writerow([
                    f"{time.time():.6f}",
                    self._cycle_id,
                    f"{run_duration_ms:.4f}",
                ])
                self._spike_file.flush()
            except Exception:
                pass

    def get_current_cycle_id(self):
        return self._cycle_id

    def close(self):
        try:
            self._main_file.close()
        except Exception:
            pass
        try:
            self._spike_file.close()
        except Exception:
            pass


def _fmt(x):
    if x is None:
        return ""
    try:
        if x != x:  # NaN
            return ""
    except Exception:
        pass
    return f"{x:.4f}"
