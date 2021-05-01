import numpy as np
import signal
import timeit
import torch.multiprocessing as mp
from typing import Dict, List

from asyreile.core.registry import WorkerRegistry
from asyreile.core.utils import set_worker_rng, get_pipeline_args_from_config

class Orchestrator(mp.Process):
  def __init__(
    self,
    registry: WorkerRegistry,
    seed_seq: np.random.SeedSequence,
    pipeline_config: List[Dict],
    max_episodes: int = 100,
  ):
    super().__init__()
    self.registry = registry
    self.seed_seq = seed_seq
    self.max_episodes = max_episodes
    self.pipeline_config = pipeline_config

  def create_pipelines(self):
    pipelines = []
    for config in self.pipeline_config:
      pipeline_args = get_pipeline_args_from_config(config)
      pipeline = pipeline_args["pipeline_class"](
        registry=self.registry,
        shared_dict=self.shared_dict,
        process_state=self.process_state,
        **pipeline_args["pipeline_config"],
      )
      pipelines.append(pipeline)

    return pipelines

  def run(self):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    start_time = timeit.default_timer()

    set_worker_rng(self.seed_seq)

    self.process_state = {
      "running": True,
      "total_steps": 0,
      "total_episodes": 0,
      "max_episodes": self.max_episodes,
    }
    self.shared_dict = {}

    pipelines = self.create_pipelines()
    for pipeline in pipelines: pipeline.start()
    for pipeline in pipelines: pipeline.join()

    print(f"{self.process_state['total_episodes']} episodes finished. {self.process_state['total_steps']} steps ran. Orchestration ended.")

    end_time = timeit.default_timer()
    duration = end_time - start_time
    print(f"Duration: {duration} seconds")
    if self.process_state['total_steps']: print(f"Average time steps per second: {duration / self.process_state['total_steps']}")

  