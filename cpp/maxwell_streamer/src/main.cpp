// maxwell_streamer —— MetaBOC Phase D：MaxOne 记录侧「只读 spike 采集探头」
// ============================================================================
// 职责（单一）：用官方 maxlab C++ DataStreamerFiltered 拿 server 侧已检测好的 spike，
//              逐行写 stdout，供 Python 端 recording_maxwell_thread 读取、聚合成
//              per-channel 计数喂 RecordingMaxwell.get_recording()。
// 只读：本程序【不触发任何刺激】。刺激由 Python 端 seq.send() 负责（见 PKM 决策
//      [[maxone-recording-phase-d-design]]）。官方示例里的 sendSequence(...) 在此被
//      「把 spike 计数经 stdout 喂回 Python」取代。
//
// IPC 协议 v1（stdout，行分隔，ASCII，空格分隔字段）：
//   # ...                                  首行 banner：版本/SDK/filter/well，'#' 开头
//   S <frameNo> <channel> <amp> <wellId>   一个 spike（server 侧检测）
//   H <frameNo>                            心跳：静默时推进 Python 端时钟，约每 --heartbeat 帧
//   数值均十进制；行尾 '\n'；每产生输出后 fflush 保证低延迟。
//   Python 侧：'S' 累计到 channel→frameNo 滚动窗口；'H'/'S' 更新当前帧号（时钟）。
//
// 退出：SIGINT/SIGTERM → 跳出循环 → DataStreamerFiltered_close()
//      （必须 close，否则 mxwserver 进入未定义态，下次实验需重启 server）。
//
// 参考（以官方为准）：
//   examples/process_spike_events.cpp（标准 filtered 闭环骨架）
//   maxlab/include/maxlab/data_streamer.h（FilteredFrameData / FrameInfo / SpikeEvent / API）
//   maxlab/include/maxlab/errors.h（Status / verifyStatus / statusToText）
// 编译：Linux 工作站 `make`（gcc>=11, gnu++20）。Mac 无 Linux libmaxlab，编不了。
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>

#include "maxlab/maxlab.h"
#include "version.h"

namespace {

std::atomic<bool> g_run{true};
void onSignal(int /*sig*/) { g_run.store(false); }

void printVersion() {
    std::printf("maxwell_streamer %s (git %s, built %s)\n",
                MAXWELL_STREAMER_VERSION, MAXWELL_STREAMER_GIT, MAXWELL_STREAMER_BUILD);
    std::printf("maxlab header  : %s (git %s)\n",
                MAXLAB_PUBLIC_API_VERSION, MAXLAB_GIT_COMMIT_HASH);
    std::printf("maxlab library : %s\n", maxlab::getStaticLibraryVersion());
}

void printHelp(const char* argv0) {
    std::printf("Usage: %s [--well N] [--filter IIR|FIR] [--heartbeat FRAMES] [--version] [--help]\n", argv0);
    std::printf("  --well N         well id to stream (default 0; MaxOne 固定 0)\n");
    std::printf("  --filter IIR|FIR filter type (default IIR, 低延迟闭环)\n");
    std::printf("  --heartbeat F    每 F 帧静默发一次 'H <frame>' 推进时钟 (default 2000 ≈100ms@20kHz; 0=关闭)\n");
    std::printf("  --version        打印版本信息并退出\n");
}

} // namespace

int main(int argc, char** argv) {
    int targetWell = 0;
    maxlab::FilterType filter = maxlab::FilterType::IIR;
    uint64_t heartbeatFrames = 2000;

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (std::strcmp(a, "--version") == 0) { printVersion(); return 0; }
        if (std::strcmp(a, "--help") == 0)    { printHelp(argv[0]); return 0; }
        if (std::strcmp(a, "--well") == 0 && i + 1 < argc) {
            targetWell = std::atoi(argv[++i]);
        } else if (std::strcmp(a, "--filter") == 0 && i + 1 < argc) {
            const char* f = argv[++i];
            if (std::strcmp(f, "FIR") == 0) filter = maxlab::FilterType::FIR;
            else if (std::strcmp(f, "IIR") == 0) filter = maxlab::FilterType::IIR;
            else { std::fprintf(stderr, "# unknown filter '%s' (use IIR or FIR)\n", f); return 2; }
        } else if (std::strcmp(a, "--heartbeat") == 0 && i + 1 < argc) {
            heartbeatFrames = std::strtoull(argv[++i], nullptr, 10);
        } else {
            std::fprintf(stderr, "# unknown arg '%s' (try --help)\n", a);
            return 2;
        }
    }

    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);

    // 版本自检：比对编译进二进制的 header 版本与链接的 lib 版本（compiled-in，不连 server）。
    maxlab::checkVersions();

    // 打开 filtered 流（spike 检测在 MaxLab Live server 侧完成）。失败 verifyStatus 会打印并 exit(1)。
    maxlab::verifyStatus(maxlab::DataStreamerFiltered_open(filter));
    std::this_thread::sleep_for(std::chrono::seconds(2)); // 等数据流稳定（官方示例同款）

    const char* filterName = (filter == maxlab::FilterType::IIR) ? "IIR" : "FIR";
    std::printf("# maxwell_streamer %s | maxlab %s(%s) | filter %s | well %d | started\n",
                MAXWELL_STREAMER_VERSION, MAXLAB_PUBLIC_API_VERSION,
                maxlab::getStaticLibraryVersion(), filterName, targetWell);
    std::fflush(stdout);

    maxlab::FilteredFrameData frame;
    uint64_t lastHeartbeat = 0;
    uint64_t noFrameStreak = 0;

    while (g_run.load()) {
        maxlab::Status st = maxlab::DataStreamerFiltered_receiveNextFrame(&frame);

        if (st == maxlab::Status::MAXLAB_NO_FRAME) {
            // 头文件：阻塞至多 1ms 仍无数据（或丢帧/坏帧）→ NO_FRAME。偶发正常；连续过多则告警。
            if (++noFrameStreak % 5000 == 0)
                std::fprintf(stderr, "# warning: %llu consecutive NO_FRAME (dropped/corrupted?)\n",
                             (unsigned long long)noFrameStreak);
            continue;
        }
        if (st != maxlab::Status::MAXLAB_OK) {
            std::fprintf(stderr, "# fatal: receiveNextFrame -> %s\n", maxlab::statusToText(st));
            break;
        }
        noFrameStreak = 0;

        if (frame.frameInfo.corrupted) continue;
        if (frame.frameInfo.well_id != static_cast<uint8_t>(targetWell)) continue;

        bool wrote = false;
        for (uint64_t i = 0; i < frame.spikeCount; ++i) {
            const maxlab::SpikeEvent& s = frame.spikeEvents[i];
            std::printf("S %llu %u %.6g %u\n",
                        (unsigned long long)s.frameNo, (unsigned)s.channel,
                        static_cast<double>(s.amp), (unsigned)s.wellId);
            wrote = true;
        }

        const uint64_t fno = frame.frameInfo.frame_number;
        if (frame.spikeCount > 0) {
            lastHeartbeat = fno; // 有 spike 已推进 Python 时钟，重置心跳基线
        } else if (heartbeatFrames > 0 && fno - lastHeartbeat >= heartbeatFrames) {
            std::printf("H %llu\n", (unsigned long long)fno);
            lastHeartbeat = fno;
            wrote = true;
        }

        if (wrote) std::fflush(stdout);
    }

    maxlab::verifyStatus(maxlab::DataStreamerFiltered_close());
    std::fprintf(stderr, "# maxwell_streamer stopped cleanly\n");
    return 0;
}
