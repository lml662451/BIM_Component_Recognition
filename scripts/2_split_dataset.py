import pandas as pd
from sklearn.model_selection import train_test_split

def stratified_split_dataset(metadata_csv: str) -> None:
    df = pd.read_csv(metadata_csv, encoding='utf-8')
    train_val, test = train_test_split(
        df,
        test_size=0.1,
        stratify=df['category'],
        random_state=42
    )
    train, val = train_test_split(
        train_val,
        test_size=0.222,
        stratify=train_val['category'],
        random_state=42
    )
    train.to_csv(r"D:\BIM_COMPONENT_RECOGNITION\metadata\train.csv", index=False, encoding='utf-8')
    val.to_csv(r"D:\BIM_COMPONENT_RECOGNITION\metadata\val.csv", index=False, encoding='utf-8')
    test.to_csv(r"D:\BIM_COMPONENT_RECOGNITION\metadata\test.csv", index=False, encoding='utf-8')

if __name__ == "__main__":
    METADATA_CSV = r"D:\BIM_COMPONENT_RECOGNITION\metadata\metadata.csv"
    stratified_split_dataset(METADATA_CSV)