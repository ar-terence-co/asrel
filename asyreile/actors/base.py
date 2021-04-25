from abc import ABC, abstractmethod
from collections import OrderedDict
import torch
from typing import Any, Dict

class BaseActor(ABC):

  @abstractmethod
  def choose_action(self, obs: torch.Tensor, greedy: bool = False) -> torch.Tensor:
    pass

  @abstractmethod
  def sync_networks(self, state_dicts: Dict[str, OrderedDict]):
    pass

  @abstractmethod
  def update(self, **kwargs):
    pass

  