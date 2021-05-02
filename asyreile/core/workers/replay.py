import torch.multiprocessing as mp
from multiprocessing.queues import Queue
import numpy as np
import signal
from threading import Thread
import time
from typing import Dict, Type

from asyreile.replays.base import BaseReplay
import asyreile.core.workers.events as events
from asyreile.core.utils import get_tensor_from_event, set_worker_rng

WARMUP_WAIT = 3 # secs

class ReplayWorker(mp.Process):
  def __init__(
    self,
    input_queue: Queue,
    buffer_queue: Queue,
    seed_seq: np.random.SeedSequence,
    replay_class: Type[BaseReplay],
    batch_size: int = 256,
    warmup_steps: int = 0,
    replay_config: Dict = {},
    index: int = 0,
    **kwargs,
  ):
    super().__init__()

    self.input_queue = input_queue
    self.buffer_queue = buffer_queue
    self.seed_seq = seed_seq

    self.replay_class = replay_class
    self.replay_config = replay_config

    self.batch_size = batch_size
    self.warmup_steps = warmup_steps

    self.index = index

  def setup(self):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    set_worker_rng(self.seed_seq)
    self.replay = self.replay_class(batch_size=self.batch_size, **self.replay_config)

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

      if task_type == events.LEARNER_ADD_EXPERIENCE_TASK:
        exp = task["experience"]
        self._store_experience(exp)

      if not self.warmup_done:
        warmup += 1
        self.warmup_done = warmup >= self.warmup_steps

  def _load_to_buffer(self):
    while True:
      if not self.warmup_done:
        time.sleep(WARMUP_WAIT)
        continue
      
      batch_exp = self.replay.sample()
      self.buffer_queue.put(batch_exp)

  def _store_experience(self, experience: Dict):
    self.replay.store(experience)
