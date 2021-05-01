from multiprocessing.queues import Empty
import numpy as np
import torch
from typing import Dict, List, Tuple

import asyreile.core.workers.events as events
from asyreile.pipelines.base import BasePipeline

class StandardObservationPipeline(BasePipeline):
  """
  Standard observation pipeline 
  """
  def __init__(
    self,
    batch_count: int = 0,
    batch_split: int = 2,
    **kwargs,
  ):
    super().__init__(**kwargs)
    
    self.env_output_queues = self.registry.output_queues["environment"]
    self.actor_input_queues = self.registry.input_queues["actor"]

    self.env_q_count = len(self.env_output_queues)
    self.actor_q_count = len(self.actor_input_queues)

    self.shared_env_output = self.env_q_count == 1
    self.batch_per_worker = self._get_batch_per_worker(batch_count, batch_split)
    self.envs_completed = [
      [False for _ in range(config["num_envs"])] 
      for config in self.registry.configs["environment"]
    ]

    self.actor_devices = [
      torch.device(
        config.get("actor_config", {}).get("device", "cuda")
      )
      for config in self.registry.configs["actor"]
    ]

  def run(self):
    actor_idx = 0

    while self.process_state["running"]:
      outputs = self._get_batch_outputs()

      if not outputs:
        if self.process_state["total_episodes"] >= self.process_state["max_episodes"]:
          self.process_state["running"] = False
        continue

      batch_obs, env_idxs = self._process_batch_outputs(outputs)
      
      task = {
        "type": events.ACTOR_CHOOSE_ACTION_TASK,
        "observation": torch.Tensor(batch_obs).to(self.actor_devices[actor_idx]),
        "greedy": False,
        "env_idx": env_idxs,
        "actor_idx": actor_idx,
      }

      self.actor_input_queues[actor_idx].put(task)
      actor_idx = (actor_idx + 1) % self.actor_q_count
      
  def _get_batch_per_worker(
    self, 
    batch_count: int = 0, 
    batch_split: int = 2,
  ) -> List[int]:
    assert batch_count > 0 or batch_split > 0

    batch_per_worker = [
      min(batch_count, config["num_envs"])
      if batch_count > 0
      else config["num_envs"] // batch_split
      for config in self.registry.configs["environment"]
    ]

    if self.shared_env_output:
      batch_per_worker = [sum(batch_per_worker)]

    return batch_per_worker

  def _is_worker_complete(self, idx: int) -> bool:
    if self.shared_env_output:
      return all(
        all(per_worker) for per_worker in self.envs_completed
      )
    return all(self.envs_completed[idx])

  def _get_batch_outputs(self) -> List[Dict]:
    outputs = []
    total_pulls = [0 for i in range(self.env_q_count)]

    while True:
      has_pull = False
      for idx in range(self.env_q_count):
        if self._is_worker_complete(idx): continue
        if total_pulls[idx] >= self.batch_per_worker[idx]: continue

        try:
          output = self.env_output_queues[idx].get(block=True, timeout=self.queue_timeout)
        except Empty:
          continue

        if output["type"] != events.RETURNED_OBSERVATION_EVENT: 
          continue # Note this skips other returned events from the env

        outputs.append(output)
        has_pull = True
        total_pulls[idx] += 1
      if not has_pull: break

    return outputs

  def _process_batch_outputs(self, outputs: List[Dict]) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    batch_obs = []
    env_idxs = []
    for output in outputs:
      batch_obs.append(output["observation"])
      env_idxs.append(output["env_idx"])
      self.process_state["total_steps"] += 1

      if output["episode_done"]:
        self.process_state["total_episodes"] += 1
        if self.process_state["total_episodes"] >= self.process_state["max_episodes"]:
          worker_idx, sub_idx = output["env_idx"]
          self.envs_completed[worker_idx][sub_idx] = True

    return batch_obs, env_idxs
