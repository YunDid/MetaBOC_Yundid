# maxwell_streamer

MetaBOC Phase D —— MaxOne 记录侧**只读 spike 采集探头**（C++）。

## 它是什么 / 为什么需要

MaxOne 的实时 spike 流官方**只能从 C++ 拿**（Python FAQ 明文 "closed-loop … not doable in Python"，maxlab Python 包无任何流式取数原语）。本程序用官方 `maxlab::DataStreamerFiltered` 拿 server 侧已检测好的 spike，逐行写 **stdout**，由 Python 端 `recording_maxwell_thread` 读取、聚合成 per-channel 计数，喂 `RecordingMaxwell.get_recording()`。

**只读**：本程序不触发任何刺激。刺激由 Python 端 `seq.send()` 负责（决策在 Python：encode_decode / robot / task）。官方示例里 spike 命中后的 `sendSequence(...)` 在此被「把 spike 计数经 stdout 喂回 Python」取代。

## 运行环境（仅 Linux 工作站）

- **gcc/g++ ≥ 13**，`-std=gnu++20`。实测：libmaxlab-1.1.0_63ce7915b 内部 ZeroMQ 由 GCC13 编译，
  引用 `GLIBCXX_3.4.31` 的 `_M_replace_cold`；g++-11/12（最高 3.4.30）链接报 undefined reference。
  Ubuntu 22.04 装 g++-13：`sudo add-apt-repository ppa:ubuntu-toolchain-r/test && sudo apt install g++-13`。
  Makefile 自动优选 g++-14/13，并用 `-static-libstdc++` 把 libstdc++ 静态打进 binary（运行时不依赖系统库版本）。
- libmaxlab（随 MaxLab SDK 发）：`~/MaxLab/share/maxlab_lib/`
  - 头文件 `maxlab/include/maxlab/*.h`，静态库 `maxlab/lib/libmaxlab.a`
  - 如未解压：`unzip ~/MaxLab/share/libmaxlab-*.zip -d ~/MaxLab/share/`
- 运行（非编译）时需 `mxwserver` 在跑 + 真机

> Mac 编不了也跑不了（Linux x86-64 静态库 + 需 mxwserver/真机）；Mac 只用于写源码。

## 构建

```bash
make                                   # → build/maxwell_streamer，并打印 --version 自检
make MAXLAB_LIB=/abs/path/maxlab_lib   # 覆盖 SDK 路径
make clean
```

链接若报 `undefined reference`：libmaxlab.a 可能有未带上的传递依赖，参照官方 `~/MaxLab/share/maxlab_lib/Makefile` 的链接行，把缺的 `-l` 追加到本 Makefile 的 `LDFLAGS`。

## 运行 / 自测

```bash
./build/maxwell_streamer --version              # 不需 server，验证链接 + 版本
./build/maxwell_streamer --well 0 --filter IIR  # 需 mxwserver + 真机；spike 行打到 stdout
./build/maxwell_streamer | head -20             # 肉眼看流
```

`Ctrl-C` / `SIGTERM` 优雅退出（会 `DataStreamerFiltered_close()`，否则 mxwserver 进入未定义态）。

## stdout 协议 v1

| 行 | 含义 |
|----|------|
| `# ...` | 首行 banner（版本/SDK/filter/well），Python 跳过或记日志 |
| `S <frameNo> <channel> <amp> <wellId>` | 一个 spike（server 侧检测）。channel 0–1023，amp 单位 bits |
| `H <frameNo>` | 心跳：静默 ≥`--heartbeat` 帧时发一次，推进 Python 端时钟 |

Python 侧：`S` 累计到 `channel → frameNo` 滚动窗口；`H`/`S` 更新当前帧号；`get_recording()` 统计窗口内每 channel 的 spike 数。刺激伪迹剔除在 Python 端按自身 stim 时刻处理（C++ 不知道 stim 时刻，保持 dumb）。

## CLI

| 参数 | 默认 | 说明 |
|------|------|------|
| `--well N` | 0 | 流哪个 well（MaxOne 固定 0） |
| `--filter IIR\|FIR` | IIR | IIR 低延迟（≈10/采样率 ms），闭环用 |
| `--heartbeat F` | 2000 | 每 F 帧静默发一次心跳（≈100ms@20kHz）；0 关闭 |
| `--version` | | 打印 maxwell_streamer + maxlab header/lib 版本后退出 |

## 版本记录

- 自身版本：`src/version.h` 的 `MAXWELL_STREAMER_VERSION`（semver，手动维护 + `CHANGELOG.md`）
- 构建期注入：git 短哈希 + 构建时间戳（Makefile `-D`）
- 链接的 SDK 版本：`--version` 同时打印 maxlab header 宏版本与 `getStaticLibraryVersion()` 实际库版本
- 当前对接：libmaxlab **1.1.0**（git `63ce7915b`）
