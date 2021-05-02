from multiprocessing.queues import Queue
import torch.multiprocessing as mp
from typing import Dict, Tuple

class WorkerRegistry:
  def __init__(self, shared = {}, **kwargs):
    self.input_queues = {}
    self.output_queues = {}
    self.buffer_queues = {}
    self.configs = {}

    self.shared = shared

  def register(
    self,
    worker_type: str, 
    config: Dict,
    maxsize: int = 0,
  ) -> Tuple[int, Queue, Queue]:
    if worker_type not in self.configs:
      self.configs[worker_type] = []
    if worker_type not in self.input_queues:
      self.input_queues[worker_type] = []
    if worker_type not in self.output_queues:
      self.output_queues[worker_type] = []

    configs = self.configs[worker_type]
    configs.append(config)
    idx = len(configs) - 1

    input_queues = self.input_queues[worker_type]
    input_queue = mp.Queue(maxsize=maxsize)
    input_queues.append(input_queue)

    output_queues = self.output_queues[worker_type]
    if self.shared.get(worker_type, False):
      if len(output_queues) == 0:
        output_queues.append(mp.Queue(maxsize=0))
      output_queue = output_queues[0]
    else:
      output_queue = mp.Queue(maxsize=maxsize)
      output_queues.append(output_queue)

    return idx, input_queue, output_queue

  def register_buffer(
    self,
    worker_type,
    maxsize: int = 0,
  ):
    if worker_type not in self.buffer_queues:
      self.buffer_queues[worker_type] = []
    
    buffer_queues = self.buffer_queues[worker_type]
    buffer_queue = mp.Queue(maxsize=maxsize)
    buffer_queues.append(buffer_queue)

    idx = len(buffer_queues) - 1

    return idx, buffer_queue
    