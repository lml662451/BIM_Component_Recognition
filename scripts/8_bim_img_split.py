import os
import shutil
import random

root_dir = r"D:\BIM_Component_Recognition\datasets"
former_dir = os.path.join(root_dir, "former")
output_dir = root_dir
split_ratio = {"train": 0.8, "val": 0.2}

for split in ["train", "val"]:
    os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)

all_images = []
for root, dirs, files in os.walk(former_dir):
    for file in files:
        if file.lower().endswith(".png"):
            all_images.append(os.path.join(root, file))

random.shuffle(all_images)
total = len(all_images)
train_end = int(total * split_ratio["train"])

train_imgs = all_images[:train_end]
val_imgs = all_images[train_end:]

def copy_images(imgs, split):
    for img_path in imgs:
        img_name = os.path.basename(img_path)
        dest_path = os.path.join(output_dir, "images", split, img_name)
        shutil.copy(img_path, dest_path)

copy_images(train_imgs, "train")
copy_images(val_imgs, "val")

print(f"训练集：{len(train_imgs)} 张")
print(f"验证集：{len(val_imgs)} 张")