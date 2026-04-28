import os
import csv

def generate_bim_metadata(root_dir: str, output_csv: str) -> None:
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["category", "instance_id", "building_code", "obj_path"])
        for category in os.listdir(root_dir):
            category_path = os.path.join(root_dir, category)
            if not os.path.isdir(category_path):
                continue
            for sub_category in os.listdir(category_path):
                sub_category_path = os.path.join(category_path, sub_category)
                if not os.path.isdir(sub_category_path):
                    continue
                for instance in os.listdir(sub_category_path):
                    instance_path = os.path.join(sub_category_path, instance)
                    if not os.path.isdir(instance_path):
                        continue
                    parts = instance.split("_")
                    building_code = parts[0]
                    instance_id = parts[-1]
                    obj_dir = os.path.join(instance_path, "OBJ")
                    if not os.path.isdir(obj_dir):
                        continue
                    obj_files = [f for f in os.listdir(obj_dir) if f.lower().endswith('.obj')]
                    if not obj_files:
                        continue
                    obj_path = os.path.join(obj_dir, obj_files[0])
                    writer.writerow([category, instance_id, building_code, obj_path])

if __name__ == "__main__":
    RAW_DATA_ROOT = r"D:\BIM_COMPONENT_RECOGNITION\raw_data"
    OUTPUT_CSV = r"D:\BIM_COMPONENT_RECOGNITION\metadata\metadata.csv"
    generate_bim_metadata(RAW_DATA_ROOT, OUTPUT_CSV)