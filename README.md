# Transformer消融实验：重现指南

## 项目概述
本项目实现了Transformer模型关键组件的消融实验，系统探究注意力头数、网络层数和嵌入维度对模型性能的影响。

## 环境要求
- **硬件最低配置**：
  - CPU：8核及以上（推荐推荐，训练速度极慢）
  - GPU：NVIDIA GPU（显存≥8GB，推荐RTX 3060及以上）
  - 内存：16GB及以上

- **软件依赖**：
  - Python 3.8~3.10
  - PyTorch 1.10+
  - Matplotlib 3.5+
  - NumPy 1.21+

## 环境配置

### 1. 创建并激活虚拟环境
```bash
# 使用conda创建环境
conda create -n transformer_ablation python=3.10
conda activate transformer_ablation

# 或使用venv
python -m venv transformer_ablation
source transformer_ablation/bin/activate  # Linux/Mac
transformer_ablation\Scripts\activate     # Windows
```

### 2. 安装依赖包
```bash
# 根据CUDA版本安装对应PyTorch（示例为CUDA 11.7）
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# 安装其他依赖
pip install matplotlib==3.6.0 numpy==1.23.5
```

## 完整实验重现步骤

### 1. 克隆代码仓库
```bash
git clone https://github.com/yourusername/transformer-ablation-study.git
cd transformer-ablation-study
```

### 2. 固定随机种子以确保结果可复现
```bash
# 设置全局随机种子（Linux/Mac）
export PYTHONHASHSEED=42
export SEED=42

# Windows系统
set PYTHONHASHSEED=42
set SEED=42
```

### 3. 运行完整消融实验
```bash
# 执行主程序（包含所有消融实验组合）
python main.py --seed $SEED  # Linux/Mac
python main.py --seed %SEED%  # Windows
```

### 4. 单独运行特定实验（可选）
```bash
# 仅运行注意力头数消融实验
python main.py --seed $SEED --experiment heads

# 仅运行网络层数消融实验
python main.py --seed $SEED --experiment layers

# 仅运行嵌入维度消融实验
python main.py --seed $SEED --experiment dims
```

## 实验输出
- 训练过程日志将实时打印到控制台
- 最终损失曲线图片保存为 `ablation_results.png`
- 实验数据以JSON格式保存到 `results/` 目录（包含各轮次训练/验证损失）

## 运行时间参考
| 硬件配置        | 完整实验运行时间 |
|-----------------|------------------|
| RTX 3090 (24GB) | 约4-6小时        |
| RTX 3060 (12GB) | 约8-10小时       |
| CPU (i7-12700)  | 约3-4天          |

## 注意事项
1. 确保GPU驱动已正确安装（CUDA版本≥11.3）
2. 若出现显存不足错误，可修改 `ablation_study()` 中的 `batch_size` 为8
3. 所有实验使用固定随机种子 `42`，确保结果可复现
4. 不同硬件平台可能存在细微性能差异，但整体趋势保持一致

通过以上步骤，您将能够完全重现本项目的所有实验结果。如有问题，请提交issue至GitHub仓库。
