from abc import ABC, abstractmethod
from collections import OrderedDict
import pathlib
import torch
from typing import Any, Dict

from asrel.core.utils import DEFAULT_CHECKPOINT_DIR

class BaseActor(ABC):
  def __init__(
    self,
    device: str = "cpu",
    global_config: Dict = {},
    **kwargs
  ):
    self.input_space = global_config["input_space"]
    self.output_space = global_config["output_space"]
    self.checkpoint_dir = pathlib.Path(global_config.get("checkpoint_dir", DEFAULT_CHECKPOINT_DIR))

    self.device = torch.device(device)

  @abstractmethod
  def choose_action(self, obs: torch.Tensor, greedy: bool = False) -> torch.Tensor:
    pass

  @abstractmethod
  def sync_networks(self, state_dicts: Dict[str, OrderedDict]):
    pass

  @abstractmethod
  def update(self, **kwargs):
    pass

  