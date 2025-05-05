import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
from src.load_cfg_file import load_config


def featurize(config):
    dataset_start_path = config["data_featurize"]["dataset_start_path"]
    dataset_target_path = config["data_featurize"]["dataset_target_path"]

    df = pd.read_csv(dataset_start_path)

    avg_Study_Hours_per_Week = df["Study_Hours_per_Week"].mean()
    delta_avg_Study_Hours_per_Week = df["Study_Hours_per_Week"] - avg_Study_Hours_per_Week

    Attendance_Rate2 = df["Attendance_Rate"] ** 2

    avg_Past_Exam_Scores = df["Past_Exam_Scores"].mean()
    delta_avg_Past_Exam_Scores = df["Past_Exam_Scores"] - avg_Past_Exam_Scores

    df["delta_avg_Study_Hours_per_Week"] = delta_avg_Study_Hours_per_Week
    df["Attendance_Rate2"] = Attendance_Rate2
    df["delta_avg_Past_Exam_Scores"] = delta_avg_Past_Exam_Scores

    df.to_csv(dataset_target_path, index=False)


if __name__ == "__main__":
    cfg = load_config("./src/config.yaml")
    featurize(cfg)