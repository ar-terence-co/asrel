import argparse
from gym import Space
import importlib
import numpy as np
import pathlib
import random
import torch
from typing import Dict, Tuple, Type
import yaml

DEFAULT_CONFIG = pathlib.Path("config.yml")


class ConfigError(Exception): pass


def get_args(description: str = "ASYnc REInforcement LEarning") -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument("--config", help="Path to the config file", default=str(DEFAULT_CONFIG))
  parser.add_argument("--test-env-worker", action="store_true", help="Run only the environment worker in the background.")
  parser.add_argument("--test-actor-worker", action="store_true", help="Run only the actor worker in the background.")
  args = parser.parse_args()
  return args


def get_config(config_file: str) -> Dict:
  with open(config_file, "r") as f:
    config = yaml.safe_load(f)
  return config


def get_seed_sequences(config: Dict):
  seed = config.get("seed")
  main_ss = np.random.SeedSequence(entropy=seed)
  print(f"SEED {main_ss.entropy}")

  env_seed_seq, actor_seed_seq, learner_seed_seq, replay_seed_seq, orchestrator_seed_seq = main_ss.spawn(5)
  return {
    "environment": env_seed_seq,
    "actor": actor_seed_seq,
    "learner": learner_seed_seq,
    "replay": replay_seed_seq,
    "orchestrator": orchestrator_seed_seq,
  }


def set_worker_rng(seed_seq: np.random.SeedSequence):
  np.random.seed(seed_seq.generate_state(4))

  torch_seed_seq, py_seed_seq = seed_seq.spawn(2)
  torch.manual_seed(torch_seed_seq.generate_state(1, dtype=np.uint64).item())

  py_seed = (py_seed_seq.generate_state(2, dtype=np.uint64).astype(object) * [1 << 64, 1]).sum()
  random.seed(py_seed)


def get_env_args_from_config(config: Dict) -> Dict:
  env_path = f"asyreile.environments.{config['path']}"
  env_class_name = config.get("class")
  env_class = get_class_from_module_path(env_path, class_name=env_class_name, class_suffix="Environment")

  return {
    "env_class": env_class,
    "num_envs": config.get("num_envs_per_worker", 2),
    "num_workers": config.get("num_workers", 1),
    "env_config": config.get("conf", {}),
  }


def get_actor_args_from_config(config: Dict) -> Dict:
  actor_path = f"asyreile.actors.{config['path']}"
  actor_class_name = config.get("class")
  print(actor_path)
  actor_class = get_class_from_module_path(actor_path, class_name=actor_class_name, class_suffix="Actor")
  print(actor_class)

  return {
    "actor_class": actor_class,
    "num_workers": config.get("num_workers", 1),
    "actor_config": config.get("conf", {})
  }


def get_net_from_config(config: Dict, input_size: Tuple[int], output_size: Tuple[int]) -> torch.nn.Module:
  net_path = f"asyreile.networks.{config['path']}"
  net_class_name = config.get("class")
  net_class = get_class_from_module_path(net_path, class_name=net_class_name, class_suffix="Network")

  net_config = config.get("conf", {})

  return net_class(**net_config, input_size=input_size, output_size=output_size)


def get_class_from_module_path(module_path, class_name: str = None, class_suffix: str = None) -> Type:
  module = importlib.import_module(module_path)
  if not class_name and class_suffix:
    for name in dir(module):
      if name.endswith(class_suffix) and name != f"Base{class_suffix}":
        class_name = name
        break
  if not class_name: raise ConfigError(f"Cannot find valid class for module `{module_path}`")
  return getattr(module, class_name)


def get_tensor_from_event(event, key):
  t = event[key].clone()
  del event[key]
  return t


def get_spaces_from_env_args(env_args: Dict) -> Tuple[Space, Space]:
  env_class = env_args["env_class"]
  env_config = env_args["env_config"]
  tmp_env = env_class(**env_config)
  observation_space = tmp_env.observation_space
  action_space = tmp_env.action_space
  return observation_space, action_space

