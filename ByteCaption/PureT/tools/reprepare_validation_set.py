#!/usr/bin/env python3
"""
重新准备 Stanford Dogs 验证集
确保图片文件名与标注品种一致
"""
import os
import json
import random
from pathlib import Path

# 配置
IMAGE_BASE_DIR = "/home/Yu_zhen/pureT/ByteCaption/PureT/stanford_dogs_jpeg"
OUTPUT_DIR = "/home/Yu_zhen/pureT/ByteCaption/PureT/data/stanford_dogs_120breeds"
VAL_SIZE = 500  # 验证集大小

# 品种名称映射（从文件夹名提取）
BREED_NAMES = {
    'n02085620': 'Chihuahua',
    'n02085782': 'Japanese spaniel',
    'n02085936': 'Maltese dog',
    'n02086079': 'Pekinese',
    'n02086240': 'Shih-Tzu',
    'n02086646': 'Blenheim spaniel',
    'n02086910': 'papillon',
    'n02087046': 'toy terrier',
    'n02087394': 'Rhodesian ridgeback',
    'n02088094': 'Afghan hound',
    'n02088238': 'basset',
    'n02088364': 'beagle',
    'n02088466': 'bloodhound',
    'n02088632': 'bluetick',
    'n02089078': 'black-and-tan coonhound',
    'n02089867': 'Walker hound',
    'n02089973': 'English foxhound',
    'n02090379': 'redbone',
    'n02090622': 'borzoi',
    'n02090721': 'Irish wolfhound',
    'n02091032': 'Italian greyhound',
    'n02091134': 'whippet',
    'n02091244': 'Ibizan hound',
    'n02091467': 'Norwegian elkhound',
    'n02091635': 'otterhound',
    'n02091831': 'Saluki',
    'n02092002': 'Scottish deerhound',
    'n02092339': 'Weimaraner',
    'n02093256': 'Staffordshire bullterrier',
    'n02093428': 'American Staffordshire terrier',
    'n02093647': 'Bedlington terrier',
    'n02093754': 'Border terrier',
    'n02093859': 'Kerry blue terrier',
    'n02093991': 'Irish terrier',
    'n02094114': 'Norfolk terrier',
    'n02094258': 'Norwich terrier',
    'n02094433': 'Yorkshire terrier',
    'n02095314': 'wire-haired fox terrier',
    'n02095570': 'Lakeland terrier',
    'n02095889': 'Sealyham terrier',
    'n02096051': 'Airedale',
    'n02096177': 'cairn',
    'n02096294': 'Australian terrier',
    'n02096437': 'Dandie Dinmont',
    'n02096585': 'Boston bull',
    'n02097047': 'miniature schnauzer',
    'n02097130': 'giant schnauzer',
    'n02097209': 'standard schnauzer',
    'n02097298': 'Scotch terrier',
    'n02097474': 'Tibetan terrier',
    'n02097658': 'silky terrier',
    'n02098105': 'soft-coated wheaten terrier',
    'n02098286': 'West Highland white terrier',
    'n02098413': 'Lhasa',
    'n02099267': 'flat-coated retriever',
    'n02099429': 'curly-coated retriever',
    'n02099601': 'golden retriever',
    'n02099712': 'Labrador retriever',
    'n02099849': 'Chesapeake Bay retriever',
    'n02100236': 'German short-haired pointer',
    'n02100583': 'vizsla',
    'n02100735': 'English setter',
    'n02100877': 'Irish setter',
    'n02101006': 'Gordon setter',
    'n02101388': 'Brittany spaniel',
    'n02101556': 'clumber',
    'n02102040': 'English springer',
    'n02102177': 'Welsh springer spaniel',
    'n02102318': 'cocker spaniel',
    'n02102480': 'Sussex spaniel',
    'n02102973': 'Irish water spaniel',
    'n02104029': 'kuvasz',
    'n02104365': 'schipperke',
    'n02105056': 'groenendael',
    'n02105162': 'malinois',
    'n02105251': 'briard',
    'n02105412': 'kelpie',
    'n02105505': 'komondor',
    'n02105641': 'Old English sheepdog',
    'n02105855': 'Shetland sheepdog',
    'n02106030': 'collie',
    'n02106166': 'Border collie',
    'n02106382': 'Bouvier des Flandres',
    'n02106550': 'Rottweiler',
    'n02106662': 'German shepherd',
    'n02107142': 'Doberman',
    'n02107312': 'miniature pinscher',
    'n02107574': 'Greater Swiss Mountain dog',
    'n02107683': 'Bernese mountain dog',
    'n02107908': 'Appenzeller',
    'n02108000': 'EntleBucher',
    'n02108089': 'boxer',
    'n02108422': 'bull mastiff',
    'n02108551': 'Tibetan mastiff',
    'n02108915': 'French bulldog',
    'n02109047': 'Great Dane',
    'n02109525': 'Saint Bernard',
    'n02109961': 'Eskimo dog',
    'n02110063': 'malamute',
    'n02110185': 'Siberian husky',
    'n02110341': 'dalmatian',
    'n02110627': 'affenpinscher',
    'n02110806': 'basenji',
    'n02110958': 'pug',
    'n02111129': 'Leonberg',
    'n02111277': 'Newfoundland',
    'n02111500': 'Great Pyrenees',
    'n02111889': 'Samoyed',
    'n02112018': 'Pomeranian',
    'n02112137': 'chow',
    'n02112350': 'keeshond',
    'n02112706': 'Brabancon griffon',
    'n02113023': 'Pembroke',
    'n02113186': 'Cardigan',
    'n02113624': 'toy poodle',
    'n02113712': 'miniature poodle',
    'n02113799': 'standard poodle',
    'n02113978': 'Mexican hairless',
    'n02114367': 'timber wolf',
    'n02114548': 'white wolf',
    'n02114712': 'dingo',
    'n02114855': 'dhole',
    'n02115049': 'African hunting dog',
}

def get_breed_name_from_folder(folder_name):
    """从文件夹名提取品种名"""
    # 文件夹格式: n02085620-Chihuahua
    parts = folder_name.split('-')
    if len(parts) >= 2:
        return '-'.join(parts[1:])  # 返回品种名部分
    # 使用映射表
    code = parts[0]
    return BREED_NAMES.get(code, code)

def main():
    print("=" * 60)
    print("重新准备 Stanford Dogs 验证集")
    print("=" * 60)

    # 收集所有图片
    all_images = []

    for split in ['test']:  # 使用 test 目录作为验证集来源
        split_dir = os.path.join(IMAGE_BASE_DIR, split)
        if not os.path.exists(split_dir):
            continue

        for folder in os.listdir(split_dir):
            folder_path = os.path.join(split_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            breed_name = get_breed_name_from_folder(folder)

            for img_file in os.listdir(folder_path):
                if img_file.endswith(('.jpg', '.JPEG')):
                    all_images.append({
                        'filename': img_file,
                        'folder': folder,
                        'breed': breed_name,
                        'split': split
                    })

    print(f"\n找到 {len(all_images)} 张图片")

    # 随机采样验证集
    random.seed(42)
    val_images = random.sample(all_images, min(VAL_SIZE, len(all_images)))

    print(f"采样 {len(val_images)} 张作为验证集")

    # 统计品种分布
    breed_counts = {}
    for img in val_images:
        breed = img['breed']
        breed_counts[breed] = breed_counts.get(breed, 0) + 1

    print(f"\n品种分布 (前10):")
    for breed, count in sorted(breed_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {breed}: {count}")

    # 生成 validation_ids.json
    validation_ids = [img['filename'].replace('.jpg', '').replace('.JPEG', '') for img in val_images]

    # 生成 annotations.json
    annotations = {
        'annotations': [
            {
                'image_id': i,
                'caption': f"The dog is a {img['breed']}."
            }
            for i, img in enumerate(val_images)
        ]
    }

    # 保存文件
    os.makedirs(os.path.join(OUTPUT_DIR, 'validation'), exist_ok=True)

    val_ids_path = os.path.join(OUTPUT_DIR, 'validation_ids.json')
    with open(val_ids_path, 'w', encoding='utf-8') as f:
        json.dump(validation_ids, f, indent=2)
    print(f"\n保存: {val_ids_path}")

    ann_path = os.path.join(OUTPUT_DIR, 'validation', 'annotations.json')
    with open(ann_path, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    print(f"保存: {ann_path}")

    # 保存完整图片信息（用于调试）
    img_info_path = os.path.join(OUTPUT_DIR, 'validation_image_info.json')
    with open(img_info_path, 'w', encoding='utf-8') as f:
        json.dump(val_images, f, indent=2, ensure_ascii=False)
    print(f"保存: {img_info_path}")

    # 验证
    print("\n" + "=" * 60)
    print("验证数据集正确性:")
    print("=" * 60)

    for i in range(min(5, len(val_images))):
        img = val_images[i]
        print(f"\n样本 {i}:")
        print(f"  文件名: {img['filename']}")
        print(f"  文件夹: {img['folder']}")
        print(f"  品种: {img['breed']}")
        print(f"  标注: {annotations['annotations'][i]['caption']}")

        # 验证图片存在
        img_path = os.path.join(IMAGE_BASE_DIR, img['split'], img['folder'], img['filename'])
        if os.path.exists(img_path):
            print(f"  图片路径: {img_path}")
        else:
            print(f"  ⚠️ 图片不存在!")

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)

if __name__ == '__main__':
    main()
