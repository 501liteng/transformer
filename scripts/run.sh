#!/bin/bash
set -e

# scripts/run.sh
# 作用：创建虚拟环境、安装依赖并运行训练入口，训练产物（曲线、表格）将写入项目根目录下的 results/。

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# Ensure results dir exists
mkdir -p "$ROOT_DIR/results"

# Create a local virtual environment in .venv if not exists
if [ ! -d ".venv" ]; then
  python -m venv .venv
fi

# Activate venv
# shellcheck source=/dev/null
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

echo "Running training. Results will be written into $ROOT_DIR/results"
export RESULTS_DIR="$ROOT_DIR/results"

# Try common entrypoints in order: src/ablation_study.py -> main.py -> src/trainer.py
if [ -f "src/ablation_study.py" ]; then
  python src/ablation_study.py
elif [ -f "main.py" ]; then
  python main.py
elif [ -f "src/trainer.py" ]; then
  python src/trainer.py
else
  echo "No training entrypoint found (src/ablation_study.py, main.py or src/trainer.py). Please run your training script manually."
  exit 1
fi
