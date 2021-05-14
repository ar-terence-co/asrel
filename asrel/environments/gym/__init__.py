from functools import partial
import gym
from typing import Dict, List, Type

from asrel.core.utils import get_class_from_module_path
from asrel.environments.base import BaseEnvironment

class GymEnvironment(gym.Wrapper, BaseEnvironment):
  def __init__(
    self, 
    id: str,
    wrappers: List[Dict], 
    **kwargs
  ):
    env = gym.make(id)
    for wrapper_config in wrappers:
      wrapper = self.get_wrapper_from_config(wrapper_config)
      env = wrapper(env)
    
    super().__init__(env)

    self.env = env

  def get_wrapper_from_config(self, config: Dict) -> Type[gym.Wrapper]:
    wrapper_path = config["path"]
    wrapper_class_name = config.get("class")
    wrapper_class = get_class_from_module_path(wrapper_path, class_name=wrapper_class_name, class_suffix="GymWrapper")
    
    return partial(wrapper_class, **config.get("conf", {}))
