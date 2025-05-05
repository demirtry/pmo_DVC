import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
from src.load_cfg_file import load_config


def preprocessing(config):
    dataset_start_path = config["data_preprocessing"]["dataset_start_path"]
    dataset_target_path = config["data_preprocessing"]["dataset_target_path"]

    df = pd.read_csv(dataset_start_path)

    columns_to_drop = [x for x in df.columns if df[x].isna().mean() > 0.15]
    columns_to_drop.append('Final_Exam_Score')
    columns_to_drop.append('Student_ID')
    df.drop(columns=columns_to_drop, inplace=True)

    df['Pass_Fail'] = df['Pass_Fail'].apply(lambda x: 1 if x == 'Pass' else 0)

    cat_columns = ['Gender', 'Parental_Education_Level', 'Internet_Access_at_Home', 'Extracurricular_Activities']
    df_encoded = pd.get_dummies(df, columns=cat_columns, drop_first=True)
    columns_to_astype = [
        "Gender_Male", "Parental_Education_Level_High School", "Parental_Education_Level_Masters",
        "Parental_Education_Level_PhD", "Internet_Access_at_Home_Yes", "Extracurricular_Activities_Yes"
    ]
    for col in columns_to_astype:
        df_encoded[col] = df_encoded[col].astype(int)

    df_encoded.to_csv(dataset_target_path)
    return True


if __name__ == "__main__":
    cfg = load_config("./src/config.yaml")
    preprocessing(cfg)