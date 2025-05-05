import sys
import os
sys.path.append(os.getcwd())

from sklearn.model_selection import train_test_split
import pandas as pd
from src.load_cfg_file import load_config


def dataset_split(config):

    df = pd.read_csv(config['data_featurize']['dataset_target_path'])
    train_dataset, test_dataset = train_test_split(df,
                                                    test_size=config['data_split']['test_size'],
                                                    random_state=42)

    train_csv_path = config['data_split']['train_path']
    test_csv_path = config['data_split']['test_path']
    train_dataset.to_csv(train_csv_path, index=False)
    test_dataset.to_csv(test_csv_path, index=False)

if __name__ == "__main__":
    cfg = load_config("./src/config.yaml")
    dataset_split(cfg)