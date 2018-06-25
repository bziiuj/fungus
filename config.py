from pathlib import Path

import yaml


def read_config():
    config_path = Path('config.yml')
    config = None
    with config_path.open('r') as f:
        config = yaml.load(f)
    return config


config = read_config()
