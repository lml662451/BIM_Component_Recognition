import os
import re
import pandas as pd

def extract_target_part(obj_file_path):
    target_str = ""
    if not os.path.exists(obj_file_path):
        return target_str
    with open(obj_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        mtl_match = re.search(r'mtllib\s+(.+)', content)
        if mtl_match:
            mtl_name = mtl_match.group(1)
            patterns = [
                r'Ifc(DuctSegment|PipeFitting)[-_](.*?)[-_][0-9]',
                r'Ifc(DuctSegment|PipeFitting)[-_](.*?)\.mtl',
                r'Ifc(DuctSegment|PipeFitting)[-_](.*)'
            ]
            for pattern in patterns:
                match = re.search(pattern, mtl_name)
                if match:
                    target_str = match.group(2)
                    break
    return target_str

def batch_extract_target(csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df['extracted_feature'] = ""
    
    for idx, row in df.iterrows():
        obj_path = row['instance_path']
        if obj_path.endswith('.obj') and os.path.exists(obj_path):
            target = extract_target_part(obj_path)
            df.loc[idx, 'extracted_feature'] = target
    
    base_name = os.path.splitext(csv_path)[0]
    new_csv_path = f"{base_name}_feature.csv"
    df.to_csv(new_csv_path, index=False, encoding='utf-8-sig')
    return new_csv_path

if __name__ == "__main__":
    CSV_INPUT = r"D:\BIM_Component_Recognition\metadata\relabel_kept.csv"
    batch_extract_target(CSV_INPUT)