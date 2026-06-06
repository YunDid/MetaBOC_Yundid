"""
MetaBOC Phase D：Maxwell 记录侧后台采集线程。

职责（单一）：以 subprocess 启动框架外 C++ 探头 maxwell_streamer，阻塞读它
stdout 的 IPC 协议 v1（S/H 行），把 spike 累计进 per-channel 的 frameNo 滚动
窗口；供 RecordingMaxwell.get_recording() 按窗口取每通道 spike 计数。

对标 Intan 的 ReadIntanDataThread + one_second 缓冲池范式（recording_intan_thread.py），
但针对 Maxwell 的两点差异做了加固：
  1. Maxwell stdout 给的是稀疏 spike 事件（不是 Intan 那种稠密逐采样波形），
     故滚动窗口用 per-channel 的 frameNo deque，而非稠密 (C, N) 数组。
  2. Intan 用 `del` 拆线程、缓冲区无锁——并发隐患。这里改为显式停止标志 +
     proc.terminate()，共享窗口读写一律加 threading.Lock。

只读：本线程不触发任何刺激。刺激由 Python 端 seq.send() 负责（与
[[MetaBOC - MaxOne 记录模块 C++ 数据流闭环接入设计]] 决策一致）。

协议 v1（与 cpp/maxwell_streamer/src/main.cpp 保持一致）：
  '# ...'                                   banner / 警告 / fatal，'#' 开头，解析时跳过
  'S <frameNo> <channel> <amp> <wellId>'    一个 spike（server 侧已检测）
  'H <frameNo>'                             心跳：静默时推进当前帧号（时钟）
"""

import os
import threading
import subprocess
from collections import deque, defaultdict


class ReadMaxwellDataThread(threading.Thread):
    """
    启动 maxwell_streamer 子进程，解析其 stdout，维护 per-channel 滚动 spike 窗口。

    用法：
        t = ReadMaxwellDataThread(binary_path, well=0)
        t.start_streamer()                 # 起子进程 + 起读取线程
        ...
        counts = t.get_channel_counts([12, 88])   # {channel: 窗口内 spike 数}
        total  = t.get_total_count([12, 88])      # 这些通道窗口内 spike 总数
        ...
        t.stop_streamer()                  # SIGTERM 子进程 + join 读取线程
    """

    def __init__(self, binary_path, well=0, fps=20000, window_sec=1.0,
                 heartbeat_frames=2000, filter_type="IIR"):
        super(ReadMaxwellDataThread, self).__init__()
        self.daemon = True  # 主进程退出时不被该线程挂住

        self.binary_path = binary_path
        self.well = int(well)
        self.fps = int(fps)
        self.window_frames = int(fps * window_sec)  # 滚动窗口长度（帧）= 默认 1s，对齐 Intan
        self.heartbeat_frames = int(heartbeat_frames)
        self.filter_type = filter_type

        self._proc = None
        self._stop_event = threading.Event()  # 不能叫 _stop：会遮蔽 Thread._stop() 内部方法
        self._lock = threading.Lock()

        # per-channel 最近 spike 的 frameNo 队列；evict 后只保留窗口内的
        self._spike_pool = defaultdict(deque)
        self._latest_frame = 0  # 当前帧号（时钟），由 S/H 行推进
        # 刺激致盲：Python 触发侧经 note_stim_frame 登记 stim 的 frameNo，
        # get_*_count 统计时剔除其后 _blank_frames 帧内的 spike（对标 Intan 剔 10ms）
        self._stim_frames = deque()
        self._blank_frames = int(fps * 0.010)  # 刺激后 10ms 致盲窗

        self._banner = None        # 记下探头首行 banner，便于诊断
        self._started_ok = False   # 见到任意 stdout 行即认为子进程在跑
        self._fatal = None         # 探头 '# fatal:' 行（若出现）

    # ------------------------------------------------------------------ 生命周期
    def start_streamer(self):
        """起子进程 + 起读取线程。binary 不存在直接抛错，不静默。"""
        if not os.path.isfile(self.binary_path):
            raise RuntimeError(
                "maxwell_streamer 二进制不存在: {}\n"
                "先在 Linux 工作站 `cd cpp/maxwell_streamer && make` 编译。".format(
                    self.binary_path
                )
            )
        cmd = [
            self.binary_path,
            "--well", str(self.well),
            "--filter", self.filter_type,
            "--heartbeat", str(self.heartbeat_frames),
        ]
        # stderr 合并进 stdout：探头的 banner/warning/fatal 都以 '#' 开头，解析时统一
        # 跳过，既避免单独 stderr 管道被写满阻塞子进程，又能在 '#' 分支顺手记诊断。
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # 行缓冲（探头每行 fflush，低延迟到达）
        )
        self.start()  # 触发 run()

    def run(self):
        """阻塞逐行读 stdout，解析 S/H，更新滚动窗口。子进程结束或被 stop 即退出。"""
        if self._proc is None or self._proc.stdout is None:
            return
        for line in self._proc.stdout:
            if self._stop_event.is_set():
                break
            line = line.strip()
            if not line:
                continue
            if line[0] == "#":
                # banner / warning / fatal —— 不进数据通路，只留诊断
                if self._banner is None:
                    self._banner = line
                self._started_ok = True
                if "fatal" in line:
                    self._fatal = line
                continue
            tok = line.split()
            tag = tok[0]
            try:
                if tag == "S" and len(tok) >= 3:
                    fn = int(tok[1])
                    ch = int(tok[2])
                    with self._lock:
                        self._latest_frame = fn
                        self._spike_pool[ch].append(fn)
                        self._evict_locked()
                elif tag == "H" and len(tok) >= 2:
                    fn = int(tok[1])
                    with self._lock:
                        self._latest_frame = fn
                        self._evict_locked()
                # 其他 tag：未知行，忽略
            except (ValueError, IndexError):
                continue  # 容错：脏行/半行跳过，不让一行坏数据掀翻线程

    def _evict_locked(self):
        """丢掉早于窗口左沿的 frameNo。调用方必须已持 _lock。"""
        floor = self._latest_frame - self.window_frames
        if floor <= 0:
            return
        for dq in self._spike_pool.values():
            while dq and dq[0] < floor:
                dq.popleft()
        while self._stim_frames and self._stim_frames[0] < floor:
            self._stim_frames.popleft()

    def stop_streamer(self):
        """优雅停：SIGTERM 子进程（探头收到后 DataStreamerFiltered_close）+ join 线程。"""
        self._stop_event.set()
        if self._proc is not None:
            try:
                self._proc.terminate()  # → 探头 SIGTERM → 优雅 close()
                self._proc.wait(timeout=3)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        if self.is_alive():
            self.join(timeout=2)  # terminate 关闭 stdout 会解除 run() 阻塞

    # ----------------------------------------------------------- 刺激致盲登记
    def note_stim_frame(self, frame_no):
        """Python 刺激触发侧在 seq.send() 前后登记该刺激的 frameNo，用于剔除刺激伪迹。

        说明：当前刺激侧尚未回传精确 frameNo（需与 mx.Event 对齐），本方法是
        Phase D 致盲接口的预留入口；未登记时 get_*_count 不做致盲，等同不剔伪迹。
        """
        with self._lock:
            self._stim_frames.append(int(frame_no))

    # --------------------------------------------------------- 消费侧（取计数）
    def get_channel_counts(self, channels=None):
        """返回 {channel: 窗口内 spike 计数}，已剔除刺激致盲窗。

        channels=None 表示统计窗口内出现过 spike 的所有通道。
        """
        with self._lock:
            blanks = [(s, s + self._blank_frames) for s in self._stim_frames]
            if channels is None:
                channels = list(self._spike_pool.keys())
            out = {}
            for ch in channels:
                dq = self._spike_pool.get(ch)
                if not dq:
                    out[ch] = 0
                    continue
                if not blanks:
                    out[ch] = len(dq)
                else:
                    out[ch] = sum(
                        1 for fn in dq
                        if not any(b0 <= fn < b1 for (b0, b1) in blanks)
                    )
            return out

    def get_total_count(self, channels=None):
        """窗口内指定通道（None=全部）的 spike 总数。"""
        return sum(self.get_channel_counts(channels).values())

    # -------------------------------------------------------------- 诊断辅助
    def is_streaming(self):
        """子进程在跑且已见到输出。"""
        return (
            self._proc is not None
            and self._proc.poll() is None
            and self._started_ok
            and self._fatal is None
        )

    @property
    def latest_frame(self):
        return self._latest_frame
