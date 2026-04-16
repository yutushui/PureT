# PureT

基于字节流的图像描述生成项目，结合 ByteFormer 和 Transformer 实现从 JPEG 压缩码流直接生成图像描述。

## 项目简介

本项目探索了一种新颖的图像描述生成方法：**直接从 JPEG 压缩码流生成描述文本**，跳过传统的图像解码步骤。

### 核心创新

- **字节流处理**：直接处理 JPEG 压缩数据，无需解码为像素
- **端到端训练**：ByteFormer 编码器 + Transformer 解码器
- **鲁棒性强**：天然抵抗图像压缩损失和传输损坏

## 应用场景：狗品种描述生成

本项目使用 **Stanford Dogs Dataset**（120 个狗品种）进行图像描述生成任务。

### 数据集

- **训练集**: 12,000 张图片
- **测试集**: 8,580 张图片
- **类别数**: 120 个狗品种
- **描述数据**: `PureT/dog_descriptions_120breeds.csv`

### 描述示例

```
The dog is a Lhasa.
It has long, fluffy fur that is predominantly white with rich brown patches
around its face and ears. The dog is standing on a polished wooden floor,
holding a blue polka-dotted toy in its mouth...
```

## 训练流程

### 两阶段训练

```
Stage 1: ByteFormer 分类预训练 (120 类狗品种分类)
    ↓
Stage 2: PureT 描述生成微调
```

### Stage 1: 分类预训练

训练 ByteFormer 学习狗品种的视觉特征：

```bash
cd ByteCaption
bash run_stage1_v2.sh
```

**配置文件**: `stanford_dogs_stage1_v2.yaml`

| 参数 | 值 |
|------|-----|
| Epochs | 80 |
| Batch Size | 24 |
| Learning Rate | 0.00015 |
| 图像尺寸 | 224×224 |
| JPEG 质量 | 90 |

**训练结果**:

| 指标 | 值 |
|------|-----|
| Top-1 准确率 | **52.27%** |
| Top-5 准确率 | **78.33%** |
| 训练时间 | ~2 小时 |

### Stage 2: 描述生成

使用 Stage 1 的权重初始化编码器，训练描述生成模型：

#### v2: 简短描述（品种+一句话）

```bash
cd ByteCaption
bash run_stage2_v2.sh
```

**配置文件**: `PureT/experiments/ByteCaption_Stage2_FirstSent/config_coco.yml`

| 参数 | 值 |
|------|-----|
| Epochs | 15 |
| Batch Size | 8 |
| Learning Rate | 0.001 |
| 词表大小 | 359 |
| 序列长度 | 60 |

**训练结果**:

| 指标 | 值 |
|------|-----|
| BLEU-1 | 85.02% |
| BLEU-4 | 78.36% |
| ROUGE-L | 86.43% |
| CIDEr | 5.24 |

#### v3: 详细描述（品种+外观+姿态+环境，~100词）

```bash
cd ByteCaption
bash run_stage2_v3_fixed.sh
```

**配置文件**: `PureT/experiments/ByteCaption_Stage2_v3/config_coco.yml`

| 参数 | 值 |
|------|-----|
| Epochs | 20 |
| Batch Size | 8 |
| Learning Rate | 0.001 |
| 词表大小 | ~800 |
| 序列长度 | 100 |

**训练结果**:

| 指标 | 值 |
|------|-----|
| 品种准确率 | **52%** (26/50) |
| CIDEr | **0.66** |
| 描述长度 | ~80-100 词 |

**预测示例**:

```
输入图片 → 模型生成描述:

"the dog is a saint bernard. it has a large, muscular build with a thick
coat that is primarily white with rich brown patches, especially around
the ears and back. the dog's face features a distinctive black mask
around its eyes and muzzle, with a broad, gentle expression. it is lying
down on a light-colored carpet, resting its head gently on its front paws,
appearing relaxed and calm. a small blue tag is visible on its collar."
```

**详细验证日志**: `results_stage2_v3/prediction_verification.log`

## 重要文件

### 训练脚本

| 文件 | 用途 |
|------|------|
| `run_stage1_v2.sh` | Stage 1 分类预训练 |
| `run_stage2_v2.sh` | Stage 2 描述生成 |
| `train_stage2_detailed.sh` | 详细版 Stage 2 训练 |

### 配置文件

| 文件 | 用途 |
|------|------|
| `stanford_dogs_stage1_v2.yaml` | Stage 1 配置 |
| `PureT/experiments/ByteCaption_Stage2_FirstSent/config_coco.yml` | Stage 2 配置 |

### 数据文件

| 文件 | 用途 |
|------|------|
| `PureT/dog_descriptions_120breeds.csv` | 120 品种狗描述数据 |
| `PureT/data/stanford_dogs_vocabulary.txt` | 词表文件 |
| `PureT/stanford_dogs_jpeg/` | 图片数据目录 |

### 训练结果

| 目录 | 内容 |
|------|------|
| `results/stanford_dogs_byteformer_stage1_v4/` | Stage 1 checkpoints |
| `results_stage2_v2/` | Stage 2 训练日志 |
| `PureT/experiments/ByteCaption_Stage2_FirstSent/snapshot/` | Stage 2 checkpoints |

## 架构

```
JPEG 字节流 → ByteFormer 编码器 → Transformer 解码器 → 图像描述
```

## 核心代码

### 模型定义 (`PureT/models/bytecaption_model.py`)

```python
class PureT_byteformer(BasicModel):
    def __init__(self):
        super(PureT_byteformer, self).__init__()
        self.vocab_size = cfg.MODEL.VOCAB_SIZE + 1

        # ByteFormer 骨干网络
        self.backbone = init_byteformer()

        # 特征嵌入
        self.att_embed = nn.Sequential(
            nn.Linear(cfg.MODEL.ATT_FEATS_DIM, cfg.MODEL.ATT_FEATS_EMBED_DIM),
            nn.ReLU(),
            nn.LayerNorm(cfg.MODEL.ATT_FEATS_EMBED_DIM),
            nn.Dropout(cfg.MODEL.DROPOUT_ATT_EMBED)
        )

        # Transformer 解码器
        self.decoder = Decoder(
            vocab_size=self.vocab_size,
            embed_dim=cfg.MODEL.BILINEAR.DIM,
            depth=cfg.MODEL.BILINEAR.DECODE_LAYERS,
            num_heads=cfg.MODEL.BILINEAR.HEAD,
            dropout=cfg.MODEL.BILINEAR.DECODE_DROPOUT
        )
```

### 编码器 (`PureT/models/encoder_decoder/PureT_encoder.py`)

```python
class Encoder(nn.Module):
    def __init__(self, embed_dim=512, depth=3, num_heads=8,
                 window_size=12, shift_size=6, use_gx=False):
        super(Encoder, self).__init__()
        # W-MSA / SW-MSA 交替结构
        self.layers = nn.ModuleList([
            EncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else shift_size,
                use_gx=use_gx
            ) for i in range(self.depth)
        ])

    def forward(self, x, att_mask=None):
        # 全局特征初始化
        gx = x.mean(1)
        if self.use_gx:
            O = torch.cat([x, gx.unsqueeze(1)], dim=1)
        for layer in self.layers:
            O = layer(O, att_mask)
        return gx, x
```

## 快速开始

```bash
# 克隆仓库（包含子模块）
git clone --recursive https://github.com/yutushui/PureT.git
cd PureT/ByteCaption

# 安装依赖
pip install -r requirements.txt

# Stage 1: 分类预训练
bash run_stage1_v2.sh

# Stage 2: 描述生成
bash run_stage2_v2.sh
```

## 监控训练

```bash
# 实时查看日志
tail -f results_stage1_v4/training.log
tail -f results_stage2_v2/training.log

# 查看训练进度
grep "Overall Training Progress" results_stage2_v2/training.log | tail -1

# 查看评估结果
grep "EVALUATION RESULTS" results_stage2_v2/training.log | tail -10
```

## 支持的模型

| 模型 | 特点 |
|------|------|
| ByteFormer + PureT | 核心方法，字节级处理 |
| BLIP | 成熟的预训练模型 |
| Qwen-VL | 多模态大模型，支持 LoRA |
| GLM-4V | 中英文双语能力 |
| InternVL | 高分辨率图像理解 |

## 评估指标

- BLEU-1/2/3/4
- METEOR
- ROUGE-L
- CIDEr
- SPICE

## 参考文献

- [ByteFormer: Bytes Are All You Need](https://arxiv.org/abs/2306.00238) - Apple
- [PureT: End-to-End Transformer Based Model for Image Captioning](https://arxiv.org/abs/2203.09912) - AAAI 2022

## 子模块

- [ByteCaption](https://github.com/YellowPerson792/ByteCaption) - 核心实现

## License

MIT License
