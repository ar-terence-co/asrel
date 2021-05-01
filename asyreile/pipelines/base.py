from abc import ABC, abstractmethod

from threading import Thread
from typing import  Callable, Dict

from asyreile.core.registry import WorkerRegistry

class BasePipeline(Thread):
  def __init__(
    self, 
    registry: WorkerRegistry,
    shared_dict: Dict,
    process_state: Dict,
    queue_timeout: int = 60,
    **kwargs,
  ):
    super().__init__()
    
    self.registry = registry
    self.shared_dict = shared_dict
    self.process_state = process_state
    self.queue_timeout = queue_timeout

  @abstractmethod
  def run(self):
    pass
