import os

root_dir = r"D:\BIM_Component_Recognition\datasets\former"

for main_category in ["IfcDuctSegment", "IfcPipeFitting"]:
    main_path = os.path.join(root_dir, main_category)
    if not os.path.isdir(main_path):
        continue
    for scene_folder in os.listdir(main_path):
        scene_path = os.path.join(main_path, scene_folder)
        if not os.path.isdir(scene_path):
            continue
        for img_file in os.listdir(scene_path):
            if not img_file.lower().endswith(".png"):
                continue
            img_id = os.path.splitext(img_file)[0]
            new_filename = f"{main_category}_{scene_folder}_{img_id}.png"
            old_file_path = os.path.join(scene_path, img_file)
            new_file_path = os.path.join(scene_path, new_filename)
            try:
                os.rename(old_file_path, new_file_path)
                print(f"{old_file_path} -> {new_file_path}")
            except Exception as e:
                print(f"{old_file_path}: {e}")

print("Done!")