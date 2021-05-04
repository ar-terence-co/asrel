from collections import OrderedDict
import torch.multiprocessing as mp
from multiprocessing.queues import Queue
import numpy as np
import signal
import torch
from typing import Dict, Iterable, List, Type

from asyreile.core.utils import set_worker_rng, get_tensor_from_event
import asyreile.core.workers.events as events
from asyreile.learners.base import BaseLearner


class LearnerWorker(mp.Process):
  def __init__(
    self,
    input_queue: Queue,
    output_queue: Queue,
    seed_seq: np.random.SeedSequence,
    learner_class: Type[BaseLearner],
    learner_config: Dict = {},
    global_config: Dict = {},
    index: int = 0,
    **kwargs,
  ):
    super().__init__()

    self.input_queue = input_queue
    self.output_queue = output_queue
    self.seed_seq = seed_seq

    self.learner_class = learner_class
    self.learner_config = learner_config

    self.global_config = global_config
    self.index = index

  def setup(self):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    print(f"Started {mp.current_process().name}.")

    set_worker_rng(self.seed_seq)
    
    self.learner = self.learner_class(
      data_stream=self.get_data_stream(),
      send_network_update=self.send_network_update,
      send_actor_update=self.send_actor_update,
      **self.learner_config, 
      global_config=self.global_config
    )

  def run(self):
    self.setup()
    self.learner.train()
    self.cleanup()

  def cleanup(self):
    print(f"Terminated {mp.current_process().name}.")

  def send_network_update(self, state_dicts: Dict[str, OrderedDict]):
    task = {
      "type": events.RETURNED_NETWORK_UPDATE_EVENT,
      "state_dicts": state_dicts,
    }
    self.output_queue.put(task)

  def send_actor_update(self, actor_params: Dict):
    task = {
      "type": events.RETURNED_ACTOR_UPDATE_EVENT,
      **actor_params,
    }
    self.output_queue.put(task)

  def get_data_stream(self) -> Iterable:
    keys = None

    while True:
      task = self.input_queue.get()
      if task["type"] == events.LEARNER_TRAIN_TASK:
        data = task["data"]
        if keys is None: keys = list(data.keys())
        tensor_data = {
          key: get_tensor_from_event(data, key) 
          if isinstance(data[key], torch.Tensor) 
          else data[key]
          for key in keys
        }

        yield tensor_data

      elif task["type"] == events.WORKER_TERMINATE_TASK:
        break

