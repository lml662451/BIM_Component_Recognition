import cv2
import numpy as np
import os

base_dataset_dir = "datasets/BIM"
output_base_dir = "datasets_lowlight/BIM"

sub_dirs = ["train", "val"]

brightness_factors = [0.3, 0.5]
noise_levels = [0, 5]

print("开始生成低光照数据...")
print(f"当前工作目录: {os.getcwd()}")
print(f"数据集根目录: {os.path.abspath(base_dataset_dir)}")

total_images = 0
generated_count = 0

for sub in sub_dirs:
    original_img_dir = os.path.join(base_dataset_dir, "images", sub)
    original_label_dir = os.path.join(base_dataset_dir, "labels", sub)
    
    output_img_dir = os.path.join(output_base_dir, "images", sub)
    output_label_dir = os.path.join(output_base_dir, "labels", sub)
    
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    print(f"\n处理 {sub} 集...")
    
    if not os.path.exists(original_img_dir):
        print(f"目录不存在，跳过")
        continue
    
    all_files = os.listdir(original_img_dir)
    img_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"找到 {len(img_files)} 张图片")
    
    for img_name in img_files:
        total_images += 1
        base = os.path.splitext(img_name)[0]
        img_path = os.path.join(original_img_dir, img_name)
        label_path = os.path.join(original_label_dir, base + '.txt')
        
        if not os.path.exists(label_path):
            print(f"跳过 {img_name}：找不到标注")
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"跳过 {img_name}：无法读取图片")
            continue
        
        with open(label_path, 'r') as f:
            label_content = f.read()
        
        for bf in brightness_factors:
            for nl in noise_levels:
                img_low = (img * bf).astype(np.uint8)
                
                if nl > 0:
                    noise = np.random.normal(0, nl, img_low.shape)
                    img_low = np.clip(img_low + noise, 0, 255).astype(np.uint8)
                
                suffix = f"bf{bf}_nl{nl}"
                new_img_name = f"{base}_{suffix}.jpg"
                new_label_name = f"{base}_{suffix}.txt"
                
                cv2.imwrite(os.path.join(output_img_dir, new_img_name), img_low)
                with open(os.path.join(output_label_dir, new_label_name), 'w') as f:
                    f.write(label_content)
                
                generated_count += 1
        
        if total_images % 10 == 0:
            print(f"已处理 {total_images} 张原始图片")

print("\n处理完成")
print(f"原始图片总数: {total_images}")
print(f"生成低光照样本数: {generated_count}")
print(f"新数据集保存在: {os.path.abspath(output_base_dir)}")