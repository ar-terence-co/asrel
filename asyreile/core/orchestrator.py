import numpy as np
import torch.multiprocessing as mp

from asyreile.core.registry import WorkerRegistry

class Orchestrator(mp.Process):
  def __init__(
    self,
    seed_seq: np.random.SeedSequence,
    registry: WorkerRegistry,
    max_episodes: int = 100,
  ):
    super().__init__()
    self.seed_seq = seed_seq
    self.registry = registry
    self.max_episodes = max_episodes

  