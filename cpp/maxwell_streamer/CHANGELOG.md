# maxwell_streamer CHANGELOG

## 0.1.0 — 2026-06-05（待 Linux 编译 + 真机验证）
- 首版骨架：`DataStreamerFiltered_open(IIR)` → 循环 `receiveNextFrame` → 按 well 过滤 → spike 逐行写 stdout（协议 v1：`S/H` 行）→ `SIGINT/SIGTERM` 优雅 `close()`。
- 照官方 `examples/process_spike_events.cpp` + `maxlab/include/maxlab/data_streamer.h` 实现；只读，不触发刺激。
- 版本记录：`version.h` semver + Makefile 注入 git/构建时间 + `--version` 打印 maxlab header/lib 版本。
- 对接 libmaxlab 1.1.0（git 63ce7915b）。
- **状态：已生成，待 Linux `make` 编译通过 + 真机流验证。**
