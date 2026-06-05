#!/usr/bin/env bash
# maxwell_cpp_link_diag.sh
# 钉死 libmaxlab.a 链接问题：①官方 Makefile（权威配方）②哪套 libstdc++ 含 _M_replace_cold
# ③可用 g++ 版本各自链哪套 libstdc++ ④conda 的 libstdc++（可能才是 MaxWell 预期的）。
# 运行（Linux 工作站，仓库根）：git pull && bash tools/maxwell_cpp_link_diag.sh
# 自动把 cpp_link_diag.txt commit+push 回来。
set -u
OUT="cpp_link_diag.txt"
SYM="_M_replace_cold"
L="$HOME/MaxLab/share/maxlab_lib"

has_sym() { nm -D "$1" 2>/dev/null | grep -q "$SYM"; }
maxglibcxx() { strings "$1" 2>/dev/null | grep -oE 'GLIBCXX_3\.4\.[0-9]+' | sort -V | tail -1; }

{
  echo "########## C++ 链接诊断 ##########"
  echo "时间 $(date)  主机 $(hostname)  内核 $(uname -r)"
  echo "CONDA_PREFIX=${CONDA_PREFIX:-<未设/已deactivate>}"
  echo

  echo "##### 1. 官方 Makefile（$L/Makefile）#####"
  if [ -f "$L/Makefile" ]; then cat "$L/Makefile"; else echo "!! 不存在"; fi
  echo
  echo "##### 1b. maxlab_lib 里其它构建线索（README/build*）#####"
  find "$L" \( -iname 'README*' -o -iname 'build*' -o -iname '*.md' \) 2>/dev/null -print -exec echo '---' \; -exec cat {} \;
  echo

  echo "##### 2. 可用编译器 + 各自链接的 libstdc++ 是否含 $SYM #####"
  for v in g++ g++-11 g++-12 g++-13 g++-14 clang++; do
    p=$(command -v "$v" 2>/dev/null) || continue
    so=$("$v" -print-file-name=libstdc++.so 2>/dev/null)
    real=$(readlink -f "$so" 2>/dev/null)
    # -print-file-name 给的是 .so（linker script/symlink），找其指向的实体或同目录 .so.6
    cand="$real"; [ -e "$cand" ] || cand=$(dirname "$so")/libstdc++.so.6
    if has_sym "$cand"; then mark="HAS $SYM"; else mark="缺 $SYM"; fi
    echo "$v ($("$v" -dumpversion 2>/dev/null)) -> $cand  [max $(maxglibcxx "$cand")]  $mark"
  done
  echo

  echo "##### 3. 系统/各处所有 libstdc++.so* 谁含 $SYM #####"
  SEARCH="/usr/lib /usr/lib/gcc /opt /home/maxwell"
  for cdir in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3" /opt/conda "${CONDA_PREFIX:-}"; do
    [ -d "$cdir" ] && SEARCH="$SEARCH $cdir"
  done
  for f in $(find $SEARCH -name 'libstdc++.so*' 2>/dev/null | xargs -r -n1 readlink -f | sort -u); do
    [ -e "$f" ] || continue
    if has_sym "$f"; then echo "HAS  $f   [max $(maxglibcxx "$f")]"; else echo "缺   $f   [max $(maxglibcxx "$f")]"; fi
  done
  echo
  echo "##### 4. 运行时默认 libstdc++.so.6（binary 跑起来会用这个）#####"
  rt=$(readlink -f /usr/lib/x86_64-linux-gnu/libstdc++.so.6 2>/dev/null)
  echo "/usr/lib/x86_64-linux-gnu/libstdc++.so.6 -> $rt  [max $(maxglibcxx "$rt")]  $(has_sym "$rt" && echo HAS || echo 缺)"
  echo "########## 诊断结束 ##########"
} > "$OUT" 2>&1

echo "已生成 $OUT （$(wc -l < "$OUT") 行）"
if git add "$OUT" && git commit -m "diag: C++ 链接诊断（_M_replace_cold 来源 / 官方 Makefile / 编译器）" \
   && git push origin "$(git rev-parse --abbrev-ref HEAD)"; then
  echo "DONE：已 push，CC 可 pull $OUT"
else
  echo "push 失败 —— 手动把 $OUT 内容贴回"
fi
