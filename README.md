# ViT Linear Probing

这个项目提供了一个简单的框架来评估视觉模型的特征表示能力。主要功能包括特征提取和线性探测评估。

## 安装

1. 克隆仓库：

```bash
git clone git@github.com:leo1oel/ViT-linear-probing.git
cd ViT-linear-probing
```

2. 安装 uv (如果还没有安装):

```bash
pip install uv
```

3. 使用 uv 创建虚拟环境并安装依赖：

```bash
uv venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows

uv pip install -r requirements.txt
```

## 使用方法

### 配置

所有配置都在 `conf/` 目录下：

- `conf/config.yaml`: 主配置文件
- `conf/model/`: 包含不同模型的具体配置

主要配置项包括：

- 数据路径配置
- 特征提取配置
- 线性探测配置
- Wandb 配置（可选）

### 运行评估

使用以下命令运行完整的评估流程：

```bash
python evaluate.py model=dino  # 使用DINO模型配置
```

如果要启用wandb记录：

```bash
python evaluate.py model=dino use_wandb=true
```

### 单独运行

也可以分别运行特征提取和线性探测：

1. 特征提取：

```bash
python extract_features.py model=dino
```

2. 线性探测：

```bash
python linear_probe.py model=dino
```

## 项目结构

```
.
├── conf/                   # 配置文件目录
│   ├── config.yaml        # 主配置文件
│   └── model/            # 模型特定配置
├── evaluate.py            # 主评估脚本
├── extract_features.py    # 特征提取脚本
├── linear_probe.py        # 线性探测脚本
├── models.py             # 模型定义
└── feature_extractor.py  # 特征提取器实现
```

## 扩展

要添加新的模型支持：

1. 在 `conf/model/` 下添加新的模型配置
2. 在 `models.py` 中实现相应的模型加载逻辑，如果是 huggingface 模型并且用最后一层 cls token 作为特征表示，可以不用修改

