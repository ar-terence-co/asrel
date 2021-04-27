import numpy as np
from threading import Thread
import torch.multiprocessing as mp
from multiprocessing.queues import Queue
import signal
from typing import Dict, List, Tuple

from asyreile.core.utils import set_worker_rng

class SimpleOrchestrator(mp.Process):
  def __init__(
    self,
    seed_seq: np.random.SeedSequence,
    **kwargs
  ):
    super().__init__()

    self.seed_seq = seed_seq
    
    self.has_shared_env_queue = kwargs.get("has_shared_env_queue", False)
    if self.has_shared_env_queue:
      self.shared_env_queue_pull_count = kwargs.get("shared_env_queue_pull_count", 1)
    else:
      self.split_env_processing = kwargs.get("split_env_processing", 2)

    self.env_input_queues = []
    self.env_output_queues = []
    self.env_counts = []

    self.actor_input_queues = []
    self.actor_output_queues = []

  def register_env_worker(self, config: Dict) -> Tuple[int, Queue, Queue]:
    num_envs = config["num_envs"]

    input_queue = mp.Queue(maxsize=num_envs)
    self.env_input_queues.append(input_queue)

    if self.has_shared_env_queue:
      if len(self.env_output_queues) == 0:
        self.env_output_queues.append(mp.Queue(maxsize=0))
      output_queue = self.env_output_queues[0]
    else:
      output_queue = mp.Queue(maxsize=num_envs)
      self.env_output_queues.append(mp.Queue(maxsize=num_envs))

    idx = len(self.env_input_queues) - 1
    self.env_counts.append(num_envs)
    return idx, input_queue, output_queue

  def register_actor_worker(self, config: Dict) -> Tuple[int, Queue, Queue]:
    input_queue = mp.Queue(maxsize=0)
    self.actor_input_queues.append(input_queue)

    if len(self.actor_output_queues) == 0:
      self.actor_output_queues.append(mp.Queue(maxsize=0))
    output_queue = self.actor_output_queues[0]

    idx = len(self.actor_input_queues) - 1
    return idx, input_queue, output_queue

  def setup(self):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    set_worker_rng(self.seed_seq)

    if self.has_shared_env_queue:
      self.env_pulls = [self.shared_env_queue_pull_count]
    else:
      self.env_pulls = [c // self.split_env_processing for c in self.env_counts]

  def run(self):
    self.setup()
    broadcast_env_outputs = Thread(target=self._broadcast_env_outputs)
    broadcast_env_outputs.start()

  def _broadcast_env_outputs(self):
    while True:
      outputs = []
      pull_counts = [0 for _ in range(len(self.env_pulls))]
      # Pull from each queue the amount in self.env_pulls
      while True:
        has_pull = False
        for i, env_output_queue in enumerate(self.env_output_queues):
          if pull_counts[i] >= self.env_pulls[i]: continue
          out = env_output_queue.get()
          print(out)
          outputs.append(out)
          has_pull = True
          pull_counts[i] += 1
        if not has_pull: break
      print(outputs)

