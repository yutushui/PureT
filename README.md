# PureT

基于字节流的图像描述生成项目，结合 ByteFormer 和 Transformer 实现从 JPEG 压缩码流直接生成图像描述。

## 项目简介

本项目探索了一种新颖的图像描述生成方法：**直接从 JPEG 压缩码流生成描述文本**，跳过传统的图像解码步骤。

### 核心创新

- **字节流处理**：直接处理 JPEG 压缩数据，无需解码为像素
- **端到端训练**：ByteFormer 编码器 + Transformer 解码器
- **鲁棒性强**：天然抵抗图像压缩损失和传输损坏

## 架构

```
JPEG 字节流 → ByteFormer 编码器 → Transformer 解码器 → 图像描述
```

## 核心代码

### 模型定义

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

### 编码器

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

# 安装依赖
pip install -r ByteCaption/requirements.txt

# 训练模型
python ByteCaption/PureT/main.py \
  --folder ByteCaption/PureT/experiments/ByteCaption_XE \
  --dataset coco \
  --load_weights \
  --freeze_backbone
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
