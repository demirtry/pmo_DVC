import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
from src.load_cfg_file import load_config


def download_data(config):
    download_path = config['data_download']['download_path']
    save_path = config['data_download']['save_path']
    df = pd.read_csv(download_path)
    df.to_csv(save_path, index=False)

    return df


if __name__ == "__main__":
    cfg = load_config("./src/config.yaml")
    download_data(cfg)