import torch.multiprocessing as mp
from multiprocessing.queues import Queue
import numpy as np
import signal
from threading import Thread
import time
from typing import Dict, Type

from asrel.core.utils import take_tensor_from_dict, set_worker_rng, validate_subclass
import asrel.core.workers.events as events
from asrel.stores.base import BaseExperienceStore

WARMUP_WAIT = 3 # secs

class ExperienceStoreWorker(mp.Process):
  def __init__(
    self,
    input_queue: Queue,
    output_queue: Queue,
    seed_seq: np.random.SeedSequence,
    store_class: Type[BaseExperienceStore],
    warmup_steps: int = 0,
    store_config: Dict = {},
    global_config: Dict = {},
    index: int = 0,
    **kwargs,
  ):
    super().__init__()

    self.input_queue = input_queue
    self.output_queue = output_queue
    self.seed_seq = seed_seq

    validate_subclass(store_class, BaseExperienceStore)
    self.store_class = store_class
    self.store_config = store_config

    self.warmup_steps = warmup_steps

    self.global_config = global_config
    self.index = index

  def setup(self):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    print(f"Started {mp.current_process().name}.")

    set_worker_rng(self.seed_seq)
    self.store = self.store_class(
      **self.store_config,
      global_config=self.global_config,
    )

    self.running = True
    self.warmup_done = False

  def start_buffer_loader(self):
    self.buffer_loader = Thread(target=self._load_to_buffer)
    self.buffer_loader.start()

  def run(self):    
    self.setup()
    self.start_buffer_loader()

    warmup = 0

    while True:
      task = self.input_queue.get()
      task_type = task["type"]

      if task_type == events.STORE_ADD_EXPERIENCE_TASK:
        exp = task["experience"]
        self._add_experience(exp)

        if not self.warmup_done:
          warmup += 1
          self.warmup_done = warmup >= self.warmup_steps
      
      elif task_type == events.WORKER_TERMINATE_TASK:
        self.running = False
        break

    self.cleanup()

  def cleanup(self):
    self.buffer_loader.join()
    print(f"Terminated {mp.current_process().name}.")

  def _load_to_buffer(self):
    while self.running:
      if not self.warmup_done:
        time.sleep(WARMUP_WAIT)
        continue
      
      batch_exp = self.store.sample()

      task = {
        "type": events.RETURNED_BATCH_DATA_EVENT,
        "data": batch_exp,
      }
      self.output_queue.put(task)

  def _add_experience(self, experience: Dict):
    self.store.add(experience)
