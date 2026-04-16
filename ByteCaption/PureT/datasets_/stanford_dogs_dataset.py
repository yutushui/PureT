"""Stanford Dogs Dataset with AI-generated captions.

Compatible with ByteCaption training pipeline.
"""

from __future__ import annotations

import os
import io
import random
import json
from typing import Any, Dict, List, Tuple, Optional, Union

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from datasets import load_from_disk
from torchvision import transforms

from PureT.lib.config import cfg
from corenet.data.transforms.jpeg_corruption import JPEGCorruptionPipeline


def pil_to_jpeg_bytes(img: Image.Image, quality: int = 60) -> bytes:
    """将PIL图像转换为JPEG字节流。"""
    if img.size != (224, 224):
        img = img.resize((224, 224), Image.BICUBIC)

    byte_buffer = io.BytesIO()
    img.save(byte_buffer, format='JPEG', quality=quality)
    byte_buffer.seek(0)
    return byte_buffer.getvalue()


class StanfordDogsDataset(data.Dataset):
    """Stanford Dogs Dataset with AI-generated captions.

    数据格式：
    - image_id: 图像ID
    - image_path: 图像文件路径
    - caption: 文字描述
    - breed: 狗品种
    - folder: 文件夹名称
    """

    def __init__(
        self,
        image_ids_path,
        input_seq,
        target_seq,
        gv_feat_path,
        seq_per_img,
        max_feat_num,
        max_samples=None,
        return_captions: bool = False,
        jpeg_quality: int = 60,
        corruption_types: Optional[List[str]] = None,
        corruption_level: str = "S0",
        corruption_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        model_type: str = "bytecaption",
        is_training: bool = True,
        save_eval_images_dir: Optional[str] = None,
        save_eval_images_max: int = 0,
    ):
        self.max_feat_num = max_feat_num
        self.seq_per_img = seq_per_img
        self.max_samples = max_samples
        self.return_captions = bool(return_captions)
        self.jpeg_quality = int(jpeg_quality)

        self.corruption_types = corruption_types or []
        self.corruption_level = corruption_level
        self.corruption_overrides = corruption_overrides or {}
        self.model_type = model_type.lower()
        self.is_training = is_training

        self.save_eval_images_dir = save_eval_images_dir
        self.save_eval_images_max = max(0, int(save_eval_images_max))
        self._saved_images = 0

        # 创建损坏管线
        pipeline_seed = int(getattr(cfg, "SEED", 0)) if getattr(cfg, "SEED", 0) > 0 else None
        self.corruption_pipeline = JPEGCorruptionPipeline(
            corruption_types=self.corruption_types,
            level=self.corruption_level,
            overrides=self.corruption_overrides,
            seed=pipeline_seed,
        ) if self.corruption_types else None

        # 加载 image_ids
        print(f"[DEBUG StanfordDogs] Loading image_ids from: {image_ids_path}")
        if image_ids_path and os.path.exists(image_ids_path):
            with open(image_ids_path, 'r', encoding='utf-8') as f:
                ids_data = json.load(f)
            if isinstance(ids_data, list):
                self.image_ids = ids_data
            else:
                self.image_ids = list(ids_data.keys())
            print(f"Loaded {len(self.image_ids)} image IDs")
        else:
            raise ValueError(f"Cannot load image_ids from {image_ids_path}")

        # 推断数据集路径和split
        basename = os.path.basename(str(image_ids_path)).lower()
        if 'val' in basename or 'valid' in basename:
            split = 'validation'
        elif 'test' in basename:
            split = 'test'
        else:
            split = 'train'

        # 数据集目录
        dataset_dir = os.path.dirname(image_ids_path)
        dataset_path = f"{dataset_dir}/{split}"

        print(f"[DEBUG StanfordDogs] Loading dataset from: {dataset_path}")
        self.ds = load_from_disk(dataset_path)

        # 创建 image_id 到数据索引的映射
        self.id_to_idx = {}
        for idx, item in enumerate(self.ds):
            self.id_to_idx[int(item['image_id'])] = idx

        # 限制样本数
        if self.max_samples and self.max_samples > 0:
            self.image_ids = self.image_ids[:self.max_samples]

        print(f"[StanfordDogs] Initialized with {len(self.image_ids)} samples")

        # 序列相关（用于非HF训练）
        self.auto_seq = False
        if not self.return_captions:
            # 这里可以添加序列加载逻辑
            self.seq_len = int(getattr(cfg.MODEL, 'SEQ_LEN', 17))

    def _load_image(self, image_path: str) -> Image.Image:
        """从文件路径加载图像"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        return Image.open(image_path).convert('RGB')

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]

        # 获取数据集中的索引
        if image_id not in self.id_to_idx:
            # 如果找不到，使用顺序索引
            data_idx = index % len(self.ds)
        else:
            data_idx = self.id_to_idx[image_id]

        item = self.ds[data_idx]
        image_path = item['image_path']
        caption = item['caption']

        # 加载图像
        img = self._load_image(image_path)

        # 如果是 visual 模型，返回 PIL 图像
        if self.model_type == "visual":
            return {
                'image_id': image_id,
                'image': img,
                'caption': caption
            }

        # ByteCaption 模型：转换为 JPEG 字节流
        jpeg_bytes = pil_to_jpeg_bytes(img, self.jpeg_quality)

        # 应用损坏（如果配置了）
        if self.corruption_pipeline and not self.is_training:
            jpeg_bytes = self.corruption_pipeline.corrupt_bytes(jpeg_bytes)

        # 返回格式与 CocoDataset 兼容
        return {
            'image_id': image_id,
            'jpeg_bytes': jpeg_bytes,
            'caption': caption
        }
