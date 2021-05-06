from multiprocessing.queues import Queue, Empty
import numpy as np
import signal
import sys
import time
import timeit
import torch.multiprocessing as mp
from typing import Dict, List

from asrel.core.registry import WorkerRegistry
from asrel.core.utils import set_worker_rng, get_pipeline_args_from_config
import asrel.core.workers.events as events

class Orchestrator(mp.Process):
  def __init__(
    self,
    registry: WorkerRegistry,
    seed_seq: np.random.SeedSequence,
    pipeline_config: List[Dict],
    global_config: Dict = {},
  ):
    super().__init__()
    self.registry = registry
    self.seed_seq = seed_seq

    self.pipeline_config = pipeline_config
    self.global_config = global_config

  def setup(self):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, self._exit)

    print(f"Started {mp.current_process().name}.")

    self.start_time = timeit.default_timer()

    set_worker_rng(self.seed_seq)

    self.process_state = {
      "running": True,
      "total_steps": 0,
      "total_episodes": 0,
      "actors_initialized": False,
    }
    self.shared_dict = {}

  def run(self):
    self.setup()

    pipelines = self._create_pipelines()
    for pipeline in pipelines: pipeline.start()
    for pipeline in pipelines: pipeline.join()

    self.cleanup()

  def cleanup(self):
    print(f"{self.process_state['total_episodes']} episodes finished. {self.process_state['total_steps']} steps ran. Orchestration ended.")

    end_time = timeit.default_timer()
    duration = end_time - self.start_time
    print(f"Duration: {duration} seconds")
    if self.process_state['total_steps']: print(f"Average time steps per second: {duration / self.process_state['total_steps']}")

    print(f"Terminating workers... Please wait until termination is complete.")
    self._terminate_workers()
    
    print(f"Terminated {mp.current_process().name}.")

  def _create_pipelines(self):
    pipelines = []
    for config in self.pipeline_config:
      pipeline_args = get_pipeline_args_from_config(config)
      pipeline = pipeline_args["pipeline_class"](
        registry=self.registry,
        shared_dict=self.shared_dict,
        process_state=self.process_state,
        **pipeline_args["pipeline_config"],
        global_config=self.global_config,
      )
      pipelines.append(pipeline)

    return pipelines

  def _terminate_workers(self):
    for key, input_queues in self.registry.input_queues.items():
      for input_queue in input_queues:
        input_queue.put({
          "type": events.WORKER_TERMINATE_TASK
        })
    
    for key, output_queues in self.registry.output_queues.items():
      for output_queue in output_queues:
        self._flush_queue(output_queue, timeout=0)
  
  def _flush_queue(self, queue: Queue, timeout: int = 0):
    try:
      while True: queue.get(block=True, timeout=timeout)
    except Empty:
      pass

  def _exit(self, signum, frame):
    self.process_state["running"] = False
    print("Terminating pipelines... Please wait until termination is complete.")
