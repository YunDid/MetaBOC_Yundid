# cpp/maxwell_streamer/Makefile
# 在 Maxwell 官方 Linux 工作站编译（gcc>=11, gnu++20）。
# Mac 编不了（无 Linux x86-64 libmaxlab.a，且需 mxwserver/真机才能运行）。
#
# 用法：
#   make                                  # 编到 build/maxwell_streamer，并打印 --version 自检
#   make MAXLAB_LIB=/abs/path/maxlab_lib  # 覆盖 SDK 路径（默认 ~/MaxLab/share/maxlab_lib）
#   make clean
#
# 链接说明：libmaxlab.a 是静态库；-lpthread 供 std::thread / 库内线程。
# 若出现 `undefined reference`，多半是 libmaxlab.a 还有传递依赖未带上——
# 参照官方 $(MAXLAB_LIB)/Makefile 的链接行，把缺的 -l 追加到下面 LDFLAGS 即可。

MAXLAB_LIB ?= $(HOME)/MaxLab/share/maxlab_lib
INCLUDE    := $(MAXLAB_LIB)/maxlab/include
LIBDIR     := $(MAXLAB_LIB)/maxlab/lib

# libmaxlab.a 内部捆绑 ZeroMQ，由 GCC>=12 编译（引用 GLIBCXX_3.4.30 的 _M_replace_cold）；
# g++ 11.x 的 libstdc++ 缺该符号会链接失败。这里自动优选 g++-13 > g++-12 > g++。
# 用 := 而非 ?= 以忽略 conda 可能注入的 CXX 环境变量；命令行 `make CXX=...` 仍可覆盖。
CXX   := $(shell command -v g++-13 2>/dev/null || command -v g++-12 2>/dev/null || echo g++)
GIT   := $(shell git rev-parse --short HEAD 2>/dev/null || echo nogit)
BUILD := $(shell date '+%Y-%m-%dT%H:%M:%S%z')

CXXFLAGS := -std=gnu++20 -O2 -Wall -Wextra -I$(INCLUDE) \
            -DMAXWELL_STREAMER_GIT=\"$(GIT)\" -DMAXWELL_STREAMER_BUILD=\"$(BUILD)\"
LDFLAGS  := -L$(LIBDIR) -lmaxlab -lpthread

TARGET := build/maxwell_streamer
SRC    := src/main.cpp src/version.h

all: $(TARGET)

$(TARGET): $(SRC)
	@test -f "$(INCLUDE)/maxlab/maxlab.h" || { echo "!! 找不到 $(INCLUDE)/maxlab/maxlab.h —— 先解压 libmaxlab 或用 make MAXLAB_LIB=... 指定"; exit 1; }
	@test -f "$(LIBDIR)/libmaxlab.a"      || { echo "!! 找不到 $(LIBDIR)/libmaxlab.a"; exit 1; }
	@mkdir -p build
	$(CXX) $(CXXFLAGS) src/main.cpp -o $(TARGET) $(LDFLAGS)
	@echo "== built: $(TARGET) =="
	@./$(TARGET) --version || true

clean:
	rm -f $(TARGET)

.PHONY: all clean
