#!/usr/bin/env bash
# maxwell_cpp_recipe_dump.sh
# 用途：libmaxlab 解压后，把 Phase D 构建所需的全部依据汇总成 cpp_build_recipe.txt 并 push，
#       供 Mac 端照真实路径/签名写 main.cpp + Makefile。包含：
#         1) maxlab_lib 目录树  2) maxlab.h / libmaxlab.a 绝对路径
#         3) 官方 Makefile/CMake/README 内容  4) 头文件内容  5) 示例 .cpp 内容
# 运行（Linux 工作站，仓库根）：git pull && bash tools/maxwell_cpp_recipe_dump.sh

set -u
L="${HOME}/MaxLab/share/maxlab_lib"
OUT="cpp_build_recipe.txt"

if [ ! -d "$L" ]; then
  echo "!! 未找到 $L —— 请先解压：unzip -o ~/MaxLab/share/libmaxlab-*.zip -d ~/MaxLab/share/"
  exit 1
fi

{
  echo "########## maxlab_lib 构建依据 dump ##########"
  echo "时间: $(date)  主机: $(hostname)  L=$L"
  echo
  echo "##### 1. maxlab_lib 完整文件树 #####"
  find "$L" | sort
  echo
  echo "##### 2. maxlab.h 与库文件绝对路径 #####"
  find "${HOME}/MaxLab" -name 'maxlab.h' 2>/dev/null
  find "${HOME}/MaxLab" \( -name 'libmaxlab*.a' -o -name 'libmaxlab*.so' \) 2>/dev/null
  echo
  echo "##### 3. 构建/说明文件内容(Makefile/CMake/README/build*.sh) #####"
  find "$L" \( -iname 'Makefile*' -o -iname 'CMakeLists*' -o -iname '*.mk' -o -iname 'README*' -o -iname '*.md' -o -iname 'build*.sh' \) 2>/dev/null \
    -print -exec echo '------------------------------------' \; -exec cat {} \; -exec echo \;
  echo
  echo "##### 4. 头文件内容(*.h / *.hpp) #####"
  find "$L" \( -name '*.h' -o -name '*.hpp' \) 2>/dev/null \
    -print -exec echo '------------------------------------' \; -exec cat {} \; -exec echo \;
  echo
  echo "##### 5. 示例源码内容(*.cpp) #####"
  find "$L" -name '*.cpp' 2>/dev/null \
    -print -exec echo '------------------------------------' \; -exec cat {} \; -exec echo \;
  echo "########## dump 结束 ##########"
} > "$OUT" 2>&1

echo "已生成 $OUT （$(wc -l < "$OUT") 行）"
echo "--- 提交并推送 ---"
if git add "$OUT" \
   && git commit -m "probe: maxlab_lib 解压后构建依据 dump（结构/路径/头文件/示例/构建文件）" \
   && git push origin "$(git rev-parse --abbrev-ref HEAD)"; then
  echo "DONE：已 push，CC 可 pull $OUT"
else
  echo "提交或 push 失败 —— 手动跑：git add $OUT && git commit -m dump && git push；或把 $OUT 内容贴回"
fi
