from abc import ABC, abstractmethod

import numpy as np
from multiprocessing.queues import Queue
import signal
from threading import Thread
import timeit
import torch.multiprocessing as mp
from typing import Dict, List, Tuple

from asyreile.core.utils import set_worker_rng

class BaseOrchestrator(mp.Process, ABC):
  def __init__(
    self,
    seed_seq: np.random.SeedSequence,
    max_episodes: int = 1000,
    **kwargs,
  ):
    super().__init__()

    self.seed_seq = seed_seq
    self.max_episodes = max_episodes

  @abstractmethod
  def register_env_worker(self, config: Dict) -> Tuple[int, Queue, Queue]:
    pass

  @abstractmethod
  def register_actor_worker(self, config: Dict) -> Tuple[int, Queue, Queue]:
    pass

  @abstractmethod
  def setup(self):
    """
    Setup after worker registtration and inside the process.
    """
    pass

  @abstractmethod
  def create_pipelines(self) -> List[Thread]:
    pass

  def run(self):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    start_time = timeit.default_timer()

    set_worker_rng(self.seed_seq)
    
    self.total_episodes = 0
    self.total_steps = 0
    self.running = True

    self.setup()

    pipelines = self.create_pipelines()
    for pipeline in pipelines: pipeline.start()
    for pipeline in pipelines: pipeline.join()

    print(f"{self.total_episodes} episodes finished. {self.total_steps} steps ran. Orchestration ended.")

    end_time = timeit.default_timer()
    duration = end_time - start_time
    print(f"Duration: {duration} seconds")
    if self.total_steps: print(f"Average time steps per second: {duration / self.total_steps}")
