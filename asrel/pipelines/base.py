from abc import ABC, abstractmethod

from multiprocessing.queues import Queue
from threading import Thread
from typing import  Callable, Dict

from asrel.core.registry import WorkerRegistry
from asrel.pipelines.utils import put_while

class BasePipeline(Thread):
  def __init__(
    self, 
    registry: WorkerRegistry,
    shared_dict: Dict,
    process_state: Dict,
    queue_timeout: int = 60,
    global_config: Dict = {},
    **kwargs,
  ):
    super().__init__()
    
    self.registry = registry
    self.shared_dict = shared_dict
    self.process_state = process_state
    self.queue_timeout = queue_timeout
    self.global_config = global_config

  def send_task(self, queue: Queue, task: Dict):
    put_while(queue, task, lambda: self.process_state["running"], timeout=self.queue_timeout)

  @abstractmethod
  def run(self):
    pass
