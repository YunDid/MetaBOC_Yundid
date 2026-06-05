# maxwell_streamer CHANGELOG

## 0.1.0 — 2026-06-05（待 Linux 编译 + 真机验证）
- 首版骨架：`DataStreamerFiltered_open(IIR)` → 循环 `receiveNextFrame` → 按 well 过滤 → spike 逐行写 stdout（协议 v1：`S/H` 行）→ `SIGINT/SIGTERM` 优雅 `close()`。
- 照官方 `examples/process_spike_events.cpp` + `maxlab/include/maxlab/data_streamer.h` 实现；只读，不触发刺激。
- 版本记录：`version.h` semver + Makefile 注入 git/构建时间 + `--version` 打印 maxlab header/lib 版本。
- 对接 libmaxlab 1.1.0（git 63ce7915b）。
- 链接修复（首次 Linux make 反馈）：去掉 `maxlab::statusToText`（errors.h 声明但未导出到 libmaxlab.a），改打数字状态码。
- 工具链定案（诊断实测）：libmaxlab.a 内部 ZeroMQ 由 **GCC13** 编译，引用 `GLIBCXX_3.4.31` 的 `_M_replace_cold`；实测该机 g++-11/12 的 libstdc++ 最高 `3.4.30`，缺该符号 → 链接失败。需装 **g++-13**（Ubuntu 22.04 走 ppa:ubuntu-toolchain-r/test）。Makefile 自动优选 g++-14/13/12，并加 `-static-libstdc++ -static-libgcc` 把 libstdc++ 静态打进 binary，使运行时（含 venv 启动）不依赖系统库版本。
- **状态：已生成，待装 g++-13 后 `make` 编译通过 + 真机流验证。**
