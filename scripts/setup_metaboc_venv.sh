#!/usr/bin/env bash
# MetaBOC Linux venv 一键配置脚本
#
# 适用场景：在 Maxwell 官方 Linux 工作站上，基于 Maxwell 自带 Python 3.10
# 创建隔离 venv，maxlab 透过 --system-site-packages 自动可见，pip install
# 写入 venv 自有目录，不污染 Maxwell 原环境。
#
# 使用方式：
#   bash scripts/setup_metaboc_venv.sh                # Stage 1（最小：numpy）
#   bash scripts/setup_metaboc_venv.sh --full         # Stage 2（完整：含 PyQt5/torch 等）
#
# 环境变量：
#   METABOC_VENV_DIR    venv 路径（默认 ~/metaboc-env）
#   MAXWELL_PYTHON      Maxwell Python（默认 /home/maxwell/MaxLab/python/bin/python3）

set -e
set -u

VENV_DIR="${METABOC_VENV_DIR:-$HOME/metaboc-env}"
MAXWELL_PYTHON="${MAXWELL_PYTHON:-/home/maxwell/MaxLab/python/bin/python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FULL_INSTALL=false

for arg in "$@"; do
    case "$arg" in
        --full) FULL_INSTALL=true ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \?//'
            exit 0
            ;;
    esac
done

banner() {
    echo ""
    echo "======================================================================"
    echo "  $1"
    echo "======================================================================"
}


# ---------- Step 1: 前置检查 ----------
banner "Step 1 / 前置检查"

if [ ! -x "$MAXWELL_PYTHON" ]; then
    echo "[FAIL] Maxwell Python not found or not executable: $MAXWELL_PYTHON"
    echo "       Set MAXWELL_PYTHON env var if your install is at a different path."
    exit 1
fi
echo "  [OK] Maxwell Python: $MAXWELL_PYTHON"
echo "       Version: $($MAXWELL_PYTHON --version)"

if ! "$MAXWELL_PYTHON" -c "import maxlab" 2>/dev/null; then
    echo "[FAIL] maxlab not importable from Maxwell Python."
    echo "       This is unexpected for a Maxwell-bundled Python. Check installation."
    exit 1
fi
echo "  [OK] maxlab visible from Maxwell Python"

echo "  Project root: $PROJECT_ROOT"
echo "  Target venv:  $VENV_DIR"


# ---------- Step 2: 创建 venv ----------
banner "Step 2 / 创建 venv（--system-site-packages）"

if [ -d "$VENV_DIR" ]; then
    echo "  [WARN] $VENV_DIR already exists."
    read -p "  Delete and recreate? [y/N] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
        "$MAXWELL_PYTHON" -m venv "$VENV_DIR" --system-site-packages
        echo "  [OK] venv recreated"
    else
        echo "  Keeping existing venv. Will install/upgrade into it."
    fi
else
    "$MAXWELL_PYTHON" -m venv "$VENV_DIR" --system-site-packages
    echo "  [OK] venv created"
fi


# ---------- Step 3: 激活 + 验证透传 ----------
banner "Step 3 / 激活 venv 与 maxlab 透传验证"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "  Active python: $(which python)"
echo "  Python version: $(python --version)"
python -c "import maxlab, sys; print('  [OK] maxlab @ ' + maxlab.__file__)"


# ---------- Step 4: 装依赖 ----------
banner "Step 4 / 安装 pip 依赖"

pip install --upgrade pip

if [ "$FULL_INSTALL" = true ]; then
    echo "  Mode: FULL (PyQt5 / torch / casadi / ...)"
    REQ_FILE="$PROJECT_ROOT/requirements_linux_py310.txt"
    if [ ! -f "$REQ_FILE" ]; then
        echo "[FAIL] $REQ_FILE not found"
        exit 1
    fi

    if command -v apt-get &> /dev/null; then
        echo "  Installing PyQt5 system libraries (sudo required)..."
        sudo apt-get update
        sudo apt-get install -y \
            libgl1-mesa-glx libegl1-mesa libxrandr2 libxss1 \
            libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6 \
            libxkbcommon-x11-0 libfontconfig1 libdbus-1-3 \
            python3-tk libglib2.0-0
    fi

    pip install -r "$REQ_FILE"
    echo "  [OK] full dependencies installed"
else
    echo "  Mode: MINIMAL (numpy only, enough for maxwell_verify.py Tier 4 + Maxwell dev)"
    pip install numpy
    echo "  [OK] minimal dependencies installed"
fi


# ---------- Step 5: 验证 ----------
banner "Step 5 / 跑 maxwell_verify.py 验证"

if [ -f "$PROJECT_ROOT/scripts/maxwell_verify.py" ]; then
    cd "$PROJECT_ROOT"
    python scripts/maxwell_verify.py || true
else
    echo "  [SKIP] scripts/maxwell_verify.py not found in $PROJECT_ROOT"
fi


banner "完成"
echo ""
echo "venv 已就绪。后续在新 terminal 用："
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "退出环境："
echo "  deactivate"
echo ""
echo "彻底删除 venv（不影响 Maxwell 原环境）："
echo "  rm -rf $VENV_DIR"
