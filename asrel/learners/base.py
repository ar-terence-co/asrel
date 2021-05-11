from abc import ABC, abstractmethod
from collections import OrderedDict
from gym import Space
import pathlib
import torch
from typing import Callable, Dict, Iterable

from asrel.core.utils import noop, set_worker_rng, DEFAULT_CHECKPOINT_DIR

class BaseLearner(ABC):
  def __init__(
    self, 
    data_stream: Iterable, 
    send_network_update: Callable[[Dict[str, OrderedDict]], None] = noop,
    send_actor_update: Callable[[Dict], None] = noop,
    device: str = "cpu",
    global_config: Dict = {},
    **kwargs
  ):
    self.data_stream = data_stream
    self._send_network_update = send_network_update
    self._send_actor_update = send_actor_update

    self.device = torch.device(device)

    self.global_config = global_config

    self.input_space = self.global_config["input_space"]
    self.output_space = self.global_config["output_space"]
    self.checkpoint_dir = pathlib.Path(self.global_config.get("checkpoint_dir", DEFAULT_CHECKPOINT_DIR))


  def send_network_update(self, state_dicts: Dict[str, OrderedDict]):
    self._send_network_update(state_dicts)
  
  def send_actor_update(self, actor_params: Dict):
    self._send_actor_update(actor_params)

  @abstractmethod
  def train(self):
    pass

  @abstractmethod
  def save_networks(self):
    pass
