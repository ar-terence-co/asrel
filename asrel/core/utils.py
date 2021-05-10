import argparse
from collections import OrderedDict
from gym import Space
import importlib
import numpy as np
import pathlib
import random
import torch
from typing import Any, Dict, Optional, Tuple, Type
import yaml

DEFAULT_CONFIG = pathlib.Path("asrel.yml")
DEFAULT_CHECKPOINT_DIR = pathlib.Path(".networks")
DEFAULT_DEVICE = torch.device("cpu")

class ConfigError(Exception): pass


def get_args(description: str = "ASync REinforcement Learning") -> argparse.Namespace:
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

  env_seed_seq, actor_seed_seq, learner_seed_seq, store_seed_seq, orchestrator_seed_seq = main_ss.spawn(5)
  return {
    "environment": env_seed_seq,
    "actor": actor_seed_seq,
    "learner": learner_seed_seq,
    "store": store_seed_seq,
    "orchestrator": orchestrator_seed_seq,
  }


def set_worker_rng(seed_seq: np.random.SeedSequence):
  np_seed_seq, torch_seed_seq, py_seed_seq = seed_seq.spawn(3)

  np.random.seed(np_seed_seq.generate_state(4))
  torch.manual_seed(torch_seed_seq.generate_state(1, dtype=np.uint64).item())
  py_seed = (py_seed_seq.generate_state(2, dtype=np.uint64).astype(object) * [1 << 64, 1]).sum()
  random.seed(py_seed)

  return np_seed_seq, torch_seed_seq, py_seed_seq

def get_registry_args_from_config(config: Dict) -> Dict:
  return {
    "shared": config.get("shared", {})
  }

def get_env_args_from_config(config: Dict) -> Dict:
  env_path = config['path']
  env_class_name = config.get("class")
  env_class = get_class_from_module_path(env_path, class_name=env_class_name, class_suffix="Environment")

  return {
    "env_class": env_class,
    "num_envs": config.get("num_envs_per_worker", 2),
    "num_workers": config.get("num_workers", 1),
    "env_config": config.get("conf", {}),
  }


def get_actor_args_from_config(config: Dict) -> Dict:
  actor_path = config['path']
  actor_class_name = config.get("class")
  actor_class = get_class_from_module_path(actor_path, class_name=actor_class_name, class_suffix="Actor")

  return {
    "actor_class": actor_class,
    "num_workers": config.get("num_workers", 1),
    "actor_config": config.get("conf", {}),
  }


def get_store_args_from_config(config: Dict) -> Dict:
  store_path = config['path']
  store_class_name = config.get("class")
  store_class = get_class_from_module_path(store_path, class_name=store_class_name, class_suffix="ExperienceStore")

  return {
    "store_class": store_class,
    "buffer_size": config.get("buffer_size", 16),
    "warmup_steps": config.get("warmup_steps", 0),
    "store_config": config.get("conf", {})
  }

def get_learner_args_from_config(config: Dict) -> Dict:
  learner_path = config['path']
  learner_class_name = config.get("class")
  learner_class = get_class_from_module_path(learner_path, class_name=learner_class_name, class_suffix="Learner")

  return {
    "learner_class": learner_class,
    "learner_config": config.get("conf", {}),
  }


def get_net_from_config(config: Dict, **kwargs) -> torch.nn.Module:
  net_path = config['path']
  net_class_name = config.get("class")
  net_class = get_class_from_module_path(net_path, class_name=net_class_name, class_suffix="Network")

  net_config = config.get("conf", {})

  return net_class(**net_config, **kwargs)


def get_orchestrator_args_from_config(config: Dict) -> Dict:
  pipeline_class_configs = []
  for c in config["pipelines"]:
    pipeline_path = c['path']
    pipeline_class_name = c.get("class")
    pipeline_class = get_class_from_module_path(pipeline_path, class_name=pipeline_class_name, class_suffix="Pipeline")
    
    pipeline_class_configs.append((pipeline_class, c.get("conf", {})))
  
  return {
    "pipeline_class_configs": pipeline_class_configs,
  }


def get_class_from_module_path(module_path, class_name: str = None, class_suffix: str = None) -> Type:
  module = importlib.import_module(module_path)
  if not class_name and class_suffix:
    for name in reversed(dir(module)):
      if name.endswith(class_suffix) and name != f"Base{class_suffix}":
        class_name = name
        break
  if not class_name: raise ConfigError(f"Cannot find valid class for module `{module_path}`")
  return getattr(module, class_name)


def take_tensor_from_dict(d: Dict[str, torch.Tensor], key: str) -> torch.Tensor:
  t = d[key].clone()
  del d[key]
  return t

def take_tensors_from_state_dicts(state_dicts: Dict[str, OrderedDict]) -> Dict[str, OrderedDict]:
  cloned_state_dicts = {}
  for net, state_dict in state_dicts.items():
    cloned_state_dicts[net] = OrderedDict()
    keys = list(state_dict.keys())
    for key in keys:
      cloned_state_dicts[net][key] = take_tensor_from_dict(state_dict, key)
  
  return cloned_state_dicts

def get_spaces_from_env_args(env_args: Dict) -> Tuple[Space, Space]:
  env_class = env_args["env_class"]
  env_config = env_args["env_config"]
  tmp_env = env_class(**env_config)
  observation_space = tmp_env.observation_space
  action_space = tmp_env.action_space
  return observation_space, action_space

def get_instance_from_config(module: Any, name: str = "", default_class: Optional[Type] = None, **kwargs) -> Any:
  try:
    class_ = getattr(module, name)
  except AttributeError:
    if not default_class: return None
    class_ = default_class
  
  return class_(**kwargs)

def validate_subclass(subclass: Type, parent: Type):
  if not issubclass(subclass, parent):
    raise ConfigError(f"{subclass.__module__}.{subclass.__name__} is not a subclass of {parent.__module__}.{parent.__name__}")

def noop(*args, **kwargs):
  pass

