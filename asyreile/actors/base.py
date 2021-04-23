from abc import ABC, abstractmethod
from typing import Any

class BaseActor(ABC):

  @abstractmethod
  def choose_action(self, obs: Any, greedy: bool = False) -> Any:
    pass

  @abstractmethod
  def sync_networks(self, network_params: Any):
    pass

  @abstractmethod
  def update(self, **kwargs):
    pass

  