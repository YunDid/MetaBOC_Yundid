#pragma once
// maxwell_streamer 自身版本（手动维护，semver）。
// 每次行为变更手动 +1，并在 CHANGELOG.md 记一行。
#define MAXWELL_STREAMER_VERSION "0.1.0"

// 以下两个由 Makefile 在编译期通过 -D 注入（git 短哈希 + 构建时间戳）。
// 未注入时给占位值，保证脱离 Makefile 直接 g++ 也能编。
#ifndef MAXWELL_STREAMER_GIT
#define MAXWELL_STREAMER_GIT "nogit"
#endif
#ifndef MAXWELL_STREAMER_BUILD
#define MAXWELL_STREAMER_BUILD "unknown"
#endif
