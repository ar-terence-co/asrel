from abc import ABC, abstractmethod
import numpy as np
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

  @abstractmethod
  def seed(self, seed: int) -> Tuple[np.random.Generator, int]:
    pass