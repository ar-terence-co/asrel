from abc import ABC, abstractmethod
from typing import Dict

class BaseExperienceStore(ABC):
  def __init__(
    self, 
    global_config: Dict = {},
    **kwargs,
  ):
    self.global_config = global_config
    self.batch_size = self.global_config.get("batch_size", 256)

  @abstractmethod
  def add(self, experience: Dict):
    pass

  @abstractmethod
  def sample(self) -> Dict:
    pass
