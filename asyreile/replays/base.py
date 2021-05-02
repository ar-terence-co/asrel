from abc import ABC, abstractmethod
from typing import Dict

class BaseReplay(ABC):
  def __init__(self, batch_size: int = 256):
    self.batch_size = batch_size

  @abstractmethod
  def store(self, experience: Dict):
    pass

  @abstractmethod
  def sample(self) -> Dict:
    pass
