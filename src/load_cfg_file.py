import yaml


def load_config(config_path):
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    return config

# print(load_config("./src/config.yaml"))
if __name__ == "__main__":
    print(load_config("config.yaml"))