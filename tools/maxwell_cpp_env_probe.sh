#!/usr/bin/env bash
# maxwell_cpp_env_probe.sh
# 用途：探测 Maxwell 官方 Linux 工作站的 C++ 构建环境，为 Phase D 的 maxwell_streamer
#       (DataStreamerFiltered → spike → IPC) binary 做准备。
# 性质：只读诊断；末尾第 5 步会把【头文件 + 示例 .cpp】(纯文本，可跨平台) 打包成一个
#       tar.gz，方便 copy 到 Mac 供 CC 照真签名写源码。不动任何系统状态、不解压、不安装。
#
# 运行（在 Linux 工作站，仓库根目录下）：
#   git pull
#   bash tools/maxwell_cpp_env_probe.sh 2>&1 | tee cpp_env_probe.out
# 然后把 cpp_env_probe.out 的内容贴回给 CC（或 git add cpp_env_probe.out && commit && push）。

MAXLAB_HOME="${HOME}/MaxLab"
LIBDIR="${MAXLAB_HOME}/share/maxlab_lib"

echo "########## Maxwell C++ 环境 probe ##########"
echo "时间: $(date)"
echo "主机: $(hostname)   架构: $(uname -m)   内核: $(uname -sr)"
echo "运行用户/HOME: $(whoami) / ${HOME}"
echo

echo "===== 1. 编译工具链（官方基线: gcc>=11, GNU C++20, GLIBC) ====="
echo "[gcc] $(gcc --version 2>/dev/null | head -1 || echo '缺失')"
echo "[g++] $(g++ --version 2>/dev/null | head -1 || echo '缺失')"
echo "[GLIBC] $(ldd --version 2>/dev/null | head -1)"
echo "[cmake] $(cmake --version 2>/dev/null | head -1 || echo '缺失(可选)')"
echo "[make] $(make --version 2>/dev/null | head -1 || echo '缺失')"
echo

echo "===== 2. libmaxlab（C++ 库，随 SDK 发，不用自建）====="
echo "[zip 包]"; ls -la "${MAXLAB_HOME}"/share/libmaxlab-*.zip 2>&1
echo "[已解压目录 maxlab_lib?]"; ls -ld "${LIBDIR}" 2>&1
echo "[maxlab.h 头文件(在 ~/MaxLab 内)]"; find "${MAXLAB_HOME}" -name 'maxlab.h' 2>/dev/null
echo "[静态/动态库 .a/.so]"; find "${MAXLAB_HOME}" \( -name 'libmaxlab*.a' -o -name 'libmaxlab*.so' \) 2>/dev/null
# 兜底：~/MaxLab 没找到头文件时，做一次有界搜索
if ! find "${MAXLAB_HOME}" -name 'maxlab.h' 2>/dev/null | grep -q .; then
  echo "[兜底搜索 maxlab.h]"; find /home /opt /usr/local -maxdepth 5 -name 'maxlab.h' 2>/dev/null | head
fi
echo

echo "===== 3. 头文件 + 示例 .cpp（可 copy 到 Mac 的文本部分）====="
if [ -d "${LIBDIR}" ]; then
  echo "[maxlab_lib 目录树(前 80 行)]"; ls -R "${LIBDIR}" 2>&1 | head -80
  echo "[所有 .h/.hpp]"; find "${LIBDIR}" \( -name '*.h' -o -name '*.hpp' \) 2>/dev/null
  echo "[所有示例 .cpp]"; find "${LIBDIR}" -name '*.cpp' 2>/dev/null
else
  echo "maxlab_lib 尚未解压。若上面 zip 存在，先解压(官方步骤)："
  echo "  unzip ${MAXLAB_HOME}/share/libmaxlab-*.zip -d ${MAXLAB_HOME}/share/"
  echo "  然后重跑本脚本。"
fi
echo

echo "===== 4. 运行期依赖(参考) ====="
echo "[mxwserver 进程]"; ps aux | grep -i mxwserver | grep -v grep | head -3
echo "[metaboc venv python]"; ls -l "${HOME}"/metaboc-env/bin/python 2>&1
echo

echo "===== 5.（可选）打包头文件+示例供 Mac 参考(只打文本，无 .a) ====="
if [ -d "${LIBDIR}" ]; then
  TAR="${HOME}/maxlab_cpp_headers_for_mac.tar.gz"
  if ( cd "${LIBDIR}" && find . \( -name '*.h' -o -name '*.hpp' -o -name '*.cpp' \) -print0 \
        | tar czf "${TAR}" --null -T - ) 2>/dev/null; then
    echo "已打包: ${TAR}"
    echo "里面是: $(tar tzf "${TAR}" 2>/dev/null | wc -l | tr -d ' ') 个 .h/.cpp 文本文件"
    echo "→ 把这个 tar.gz copy 到 Mac，解压到 Maxwell-Yundid/Code/maxlab_cpp/ 即可。"
  else
    echo "打包失败(无头文件或权限)；可手动 copy ${LIBDIR} 下的 *.h / *.cpp 到 Mac。"
  fi
else
  echo "(maxlab_lib 未解压，跳过打包)"
fi
echo
echo "########## probe 结束。请把 cpp_env_probe.out 内容贴回 ##########"
