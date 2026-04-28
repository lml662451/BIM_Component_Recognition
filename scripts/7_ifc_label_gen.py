import pandas as pd
import os

INPUT_CSV = r"D:\BIM_Component_Recognition\metadata\relabel_kept_feature.csv"
OUTPUT_CSV = r"D:\BIM_Component_Recognition\metadata\relabel_kept_feature_with_label.csv"

def get_train_label(feature_str, main_class):
    if pd.isna(feature_str):
        return f"{main_class}_未知类别"
    
    feature_str = str(feature_str).strip()
    feature_clean = (
        feature_str
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
        .lower()
    )
    main_class = str(main_class).strip()

    if main_class == "IfcDuctSegment":
        if "矩形风管" in feature_str:
            return "IfcDuctSegment_矩形风管"
        elif "圆形风管" in feature_str:
            return "IfcDuctSegment_圆形风管"
        else:
            return "IfcDuctSegment_其他风管"
    elif main_class == "IfcPipeFitting":
        if "同心变径管接头" in feature_str or "同心变径" in feature_clean:
            return "IfcPipeFitting_同心变径管接头"
        elif "T形三通" in feature_str or "t形三通" in feature_clean:
            return "IfcPipeFitting_T形三通"
        elif "变径三通" in feature_str or "变径三通" in feature_clean:
            return "IfcPipeFitting_变径三通"
        elif "异径三通" in feature_str or "异径三通" in feature_clean:
            return "IfcPipeFitting_异径三通"
        elif "变径四通" in feature_str or "变径四通" in feature_clean:
            return "IfcPipeFitting_变径四通"
        elif "三通" in feature_str:
            return "IfcPipeFitting_三通"
        elif "四通" in feature_str:
            return "IfcPipeFitting_四通"
        elif "弯头" in feature_str:
            return "IfcPipeFitting_弯头类"
        elif "过渡件" in feature_str:
            return "IfcPipeFitting_过渡件"
        else:
            return "IfcPipeFitting_其他管件"
    else:
        return f"{main_class}_未知类别"

if __name__ == "__main__":
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"源文件不存在：{INPUT_CSV}")
    
    try:
        df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    except:
        df = pd.read_csv(INPUT_CSV, encoding="gbk")
    
    last_col_name = df.columns[-1]
    main_class_col = df.columns[1]
    
    df["train_label"] = df.apply(lambda row: get_train_label(row[last_col_name], row[main_class_col]), axis=1)
    predefined_classes = [
        "IfcDuctSegment_矩形风管",
        "IfcDuctSegment_圆形风管",
        "IfcDuctSegment_其他风管",
        "IfcPipeFitting_T形三通",
        "IfcPipeFitting_变径三通",
        "IfcPipeFitting_异径三通",
        "IfcPipeFitting_变径四通",
        "IfcPipeFitting_三通",
        "IfcPipeFitting_四通",
        "IfcPipeFitting_同心变径管接头",
        "IfcPipeFitting_弯头类",
        "IfcPipeFitting_过渡件",
        "IfcPipeFitting_其他管件"
    ]
    label2id = {label: idx for idx, label in enumerate(predefined_classes)}
    df["train_label_id"] = df["train_label"].map(lambda x: label2id.get(x, -1))
    
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"结果已保存至新文件：{OUTPUT_CSV}")