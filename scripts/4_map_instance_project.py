import pandas as pd
import os

def extract_identifier_from_instance_path(path):
    return os.path.basename(path)

def extract_identifier_from_obj_path(path):
    return os.path.basename(os.path.dirname(os.path.dirname(path)))

def main():
    relabel_path = r"D:\BIM_Component_Recognition\metadata\relabel_filtered.csv"
    train_path = r"D:\BIM_Component_Recognition\metadata\train_filtered.csv"
    output_path = r"D:\BIM_Component_Recognition\metadata\relabel_kept.csv"

    relabel_df = pd.read_csv(relabel_path)
    train_df = pd.read_csv(train_path)

    relabel_df['identifier'] = relabel_df['instance_path'].apply(extract_identifier_from_instance_path)
    train_df['identifier'] = train_df['obj_path'].apply(extract_identifier_from_obj_path)

    train_map = train_df[['identifier', 'obj_path']].drop_duplicates(subset=['identifier'])
    relabel_merged = pd.merge(relabel_df, train_map, on='identifier', how='inner')

    relabel_merged['instance_path'] = relabel_merged['obj_path']
    final_df = relabel_merged.drop(columns=['obj_path'])

    final_df.to_csv(output_path, index=False, encoding='utf-8')

if __name__ == "__main__":
    main()