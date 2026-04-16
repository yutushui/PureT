import os
import json
from PIL import Image

# 导入工厂模式
from PureT.models import create

# ----------------------------
# 配置路径
# ----------------------------
IMG_DIR = "PureT/data/coco_karpathy/test_sample_500"
CAPTIONS_FILE = "PureT/data/coco_karpathy/captions_test.json"

# ----------------------------
# 加载图片列表
# ----------------------------
with open(CAPTIONS_FILE, "r") as f:
    captions_data = json.load(f)

# 确保 captions_data 是 dict
if isinstance(captions_data, list):
    captions_data = {str(i): item for i, item in enumerate(captions_data)}

# ----------------------------
# 创建模型
# ----------------------------
model = create("PureT_byteformer")  # 工厂模式创建模型

# 如果模型需要加载权重，可在这里加：
# model.load_state_dict(torch.load("path_to_weights.pth"))

# ----------------------------
# 遍历图片，生成描述
# ----------------------------
for idx, image_id in captions_data.items():
    img_path = os.path.join(IMG_DIR, captions_data[idx])
    
    if not os.path.exists(img_path):
        print(f"[WARN] 图片不存在: {img_path}")
        continue

    # 打开图片
    img = Image.open(img_path).convert("RGB")
    
    # 调用模型生成 caption（假设模型有一个 generate 方法）
    try:
        caption = model.generate(img)  # 注意：根据你的模型修改接口
    except Exception as e:
        caption = f"[ERROR] {e}"

    print(f"{image_id} -> {caption}")