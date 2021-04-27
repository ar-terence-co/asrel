from abc import ABC, abstractmethod
from collections import OrderedDict
from gym import Space
import torch
from typing import Any, Dict

class BaseActor(ABC):
  def __init__(
    self,
    input_space: Space,
    output_space: Space,
  ):
    self.input_space = input_space
    self.output_space = output_space

  @abstractmethod
  def choose_action(self, obs: torch.Tensor, greedy: bool = False) -> torch.Tensor:
    pass

  @abstractmethod
  def sync_networks(self, state_dicts: Dict[str, OrderedDict]):
    pass

  @abstractmethod
  def update(self, **kwargs):
    pass

  