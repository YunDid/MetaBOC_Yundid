#!/usr/bin/env bash
# MetaBOC Linux 一键环境配置脚本
#
# 适用场景：在 Maxwell 官方 Linux 工作站上配置 MetaBOC 完整运行环境
# 前提：已安装 Miniconda 或 Anaconda，conda 命令可用
#
# 使用方式：
#   bash scripts/setup_metaboc_linux.sh
#   或赋权后直接执行：
#   chmod +x scripts/setup_metaboc_linux.sh && ./scripts/setup_metaboc_linux.sh
#
# 依据：[[MetaBOC - Linux 环境配置与 Python 3.6 依赖版本锁定]]

set -e   # 任一命令失败即退出
set -u   # 引用未定义变量即失败

ENV_NAME="${METABOC_ENV_NAME:-metaboc}"
PYTHON_VERSION="3.6"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REQUIREMENTS_FILE="$PROJECT_ROOT/requirements_linux.txt"


banner() {
    echo ""
    echo "======================================================================"
    echo "  $1"
    echo "======================================================================"
}


check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "[FAIL] Required command not found: $1"
        return 1
    fi
}


# ---------- Step 1: 前置检查 ----------
banner "Step 1 / 前置检查"

check_command conda || {
    echo "Please install Miniconda first:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
}
echo "  [OK] conda found: $(conda --version)"

if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "[FAIL] requirements file not found: $REQUIREMENTS_FILE"
    exit 1
fi
echo "  [OK] requirements file: $REQUIREMENTS_FILE"


# ---------- Step 2: 系统级依赖（PyQt5 底层库） ----------
banner "Step 2 / 系统级依赖（需要 sudo）"

if command -v apt-get &> /dev/null; then
    echo "  Detected apt-get. Will install PyQt5 system libraries."
    echo "  You may be prompted for sudo password."
    sudo apt-get update
    sudo apt-get install -y \
        libgl1-mesa-glx libegl1-mesa libxrandr2 libxss1 \
        libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6 \
        libxkbcommon-x11-0 libfontconfig1 libdbus-1-3 \
        python3-tk libglib2.0-0
    echo "  [OK] system libraries installed"
else
    echo "  [WARN] apt-get not found. Skipping system library install."
    echo "         If PyQt5 fails to load, install equivalent packages for your distro."
fi


# ---------- Step 3: 创建 conda env ----------
banner "Step 3 / 创建 conda 环境（$ENV_NAME, python=$PYTHON_VERSION）"

if conda env list | grep -q "^${ENV_NAME}\s"; then
    echo "  [WARN] Environment '$ENV_NAME' already exists."
    read -p "  Recreate it? This will delete the existing env. [y/N] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n "$ENV_NAME" -y
        conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
        echo "  [OK] env recreated"
    else
        echo "  Keeping existing env. Will install/upgrade packages into it."
    fi
else
    conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
    echo "  [OK] env created"
fi


# ---------- Step 4: 激活 env 并安装 pip 依赖 ----------
banner "Step 4 / 安装 pip 依赖"

# 必须 source conda.sh 才能在脚本内使用 conda activate
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "  Active python: $(which python)"
echo "  Python version: $(python --version)"

pip install --upgrade pip
pip install -r "$REQUIREMENTS_FILE"
echo "  [OK] pip dependencies installed"


# ---------- Step 5: maxlab 链接（系统自带 → conda env） ----------
banner "Step 5 / 链接系统 maxlab 到 conda env"

echo "  Maxwell 官方 Linux 机器自带 maxlab，但默认只对系统 python 可见。"
echo "  让 conda env 也能 import maxlab，有三种方案："
echo ""
echo "  方案 A（推荐，无侵入）：每次激活 env 时通过 PYTHONPATH 注入"
echo "    1. 找到系统 maxlab 路径："
echo "       /usr/bin/python3 -c 'import maxlab; import os; print(os.path.dirname(os.path.dirname(maxlab.__file__)))'"
echo "    2. 创建 conda env 激活钩子："
echo "       mkdir -p \$CONDA_PREFIX/etc/conda/activate.d"
echo "       echo 'export PYTHONPATH=<上面打印的路径>:\$PYTHONPATH' \\"
echo "         > \$CONDA_PREFIX/etc/conda/activate.d/maxlab.sh"
echo ""
echo "  方案 B：在 env 的 site-packages 里软链 maxlab"
echo "    SITE_PKG=\$(python -c 'import sysconfig; print(sysconfig.get_paths()[\"purelib\"])')"
echo "    ln -s <系统 maxlab 路径> \$SITE_PKG/maxlab"
echo ""
echo "  方案 C：在 maxwell_verify.py 等脚本运行前 PYTHONPATH=<...> 临时注入"
echo ""
echo "  本脚本不自动执行（需要你确认系统 maxlab 路径）。"


# ---------- Step 6: 验证 ----------
banner "Step 6 / 基础验证"

python -c "import PyQt5; print('  [OK] PyQt5')"
python -c "import torch; print('  [OK] PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python -c "import numpy; print('  [OK] NumPy:', numpy.__version__)"
python -c "import casadi; print('  [OK] CasADi')"

echo ""
echo "  尝试 import maxlab（链接成功才有结果）："
python -c "import maxlab; print('  [OK] maxlab @', maxlab.__file__)" 2>/dev/null || \
    echo "  [SKIP] maxlab not visible in conda env. See Step 5 for linking options."


banner "完成"
echo ""
echo "下一步："
echo "  1. 激活环境：    conda activate $ENV_NAME"
echo "  2. 跑验证脚本：   python scripts/maxwell_verify.py"
echo "  3. 跑主程序：    python main.py"
echo ""
echo "注意：当前 MetaBOC_Yundid 代码尚未做 Linux 适配（MCS clr 链顶层导入）。"
echo "      main.py 在 Linux 上运行需要先实现 platform_config 条件分支改造。"
echo "      maxwell_verify.py 不依赖 communication.py，可独立运行。"
