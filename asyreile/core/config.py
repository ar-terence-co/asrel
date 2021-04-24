from typing import Dict
import yaml


_config = {}


def load_config(config_file: str):
  global _config

  with open(config_file, "r") as f:
    _config = yaml.safe_load(f)
  

def get_config() -> Dict:
  return _config
