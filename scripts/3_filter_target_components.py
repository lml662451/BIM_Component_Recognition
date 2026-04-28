import pandas as pd

relabel_df = pd.read_csv(r"D:\BIM_Component_Recognition\metadata\relabel_component_index.csv", encoding='utf-8')
train_df = pd.read_csv(r"D:\BIM_Component_Recognition\metadata\train.csv", encoding='utf-8')

target_components = ["IfcDuctSegment", "IfcPipeFitting"]
relabel_filtered = relabel_df[relabel_df['component_type'].isin(target_components)]
train_filtered = train_df[train_df['category'].isin(target_components)]

relabel_filtered.to_csv(r"D:\BIM_Component_Recognition\metadata\relabel_filtered.csv", index=False, encoding='utf-8')
train_filtered.to_csv(r"D:\BIM_Component_Recognition\metadata\train_filtered.csv", index=False, encoding='utf-8')