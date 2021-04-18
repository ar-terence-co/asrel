from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

class BaseEnvironment(ABC):
  def __init__(self):
    self.observation_space = None
    self.action_space = None

  @abstractmethod
  def reset(self) -> Any:
    pass

  @abstractmethod
  def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
    pass