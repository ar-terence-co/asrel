from multiprocessing.queues import Empty
import numpy as np
import time
import torch
from typing import Dict, List, Tuple

import asrel.core.workers.events as events
from asrel.pipelines.base import BasePipeline

ACTOR_WAIT_TIMEOUT = 3 # seconds
MOVING_AVERAGE_STEPS = 100 # steps
SAVE_IF_IDLE = 100 # episodes

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
    self.store_input_queue = self.registry.input_queues["store"][0]
    self.learner_input_queue = self.registry.input_queues["learner"][0]

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

    if "experiences" not in self.shared_dict:
      self.shared_dict["experiences"] = {}
    if "scores" not in self.shared_dict:
      self.shared_dict["scores"] = np.array([], dtype=float)

    self.max_episodes = self.global_config.get("max_episodes", 100)
    self.n_steps = self.global_config.get("n_steps", 1)
    self.gamma = self.global_config.get("gamma", 1.0)

    self.last_saved = -1

  def run(self):
    self._wait_for_actors()

    actor_idx = 0

    while self.process_state["running"]:
      outputs = self._get_batch_outputs()

      if not outputs:
        if self.process_state["total_episodes"] >= self.max_episodes:
          self.process_state["running"] = False
        continue

      for output in outputs:
        to_store = self._update_experiences(output)
        for exp in to_store:
          task = {
            "type": events.STORE_ADD_EXPERIENCE_TASK,
            "experience": exp,
          }
          self.send_task(self.store_input_queue, task)

      batch_obs, env_idxs, should_save = self._process_batch_outputs(outputs)
      task = {
        "type": events.ACTOR_CHOOSE_ACTION_TASK,
        "observation": torch.tensor(batch_obs).to(self.actor_devices[actor_idx]),
        "greedy": False,
        "env_idx": env_idxs,
        "actor_idx": actor_idx,
      }
      self.send_task(self.actor_input_queues[actor_idx], task)

      if should_save:
        task = {
          "type": events.LEARNER_SAVE_NETWORKS_TASK
        }
        self.send_task(self.learner_input_queue, task)
        self.last_saved = self.process_state["total_episodes"]

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
    should_save = False
    for output in outputs:
      batch_obs.append(output["observation"])
      env_idxs.append(output["env_idx"])
      self.process_state["total_steps"] += 1

      if output["episode_done"]:
        self.process_state["total_episodes"] += 1
        self.shared_dict["scores"] = np.append(self.shared_dict["scores"], output["score"])
        average_score = self.shared_dict["scores"][-MOVING_AVERAGE_STEPS:].mean()
        if (
          self.process_state["total_episodes"] - self.last_saved > SAVE_IF_IDLE or
          "max_average_score" not in self.shared_dict or 
          average_score > self.shared_dict["max_average_score"]
        ):
          self.shared_dict["max_average_score"] = average_score
          should_save = True

        print(
          self.process_state["total_episodes"],
          output["env_idx"],
          output["episode_step"],
          output["score"],
          average_score,
        )

        if self.process_state["total_episodes"] >= self.max_episodes:
          worker_idx, sub_idx = output["env_idx"]
          self.envs_completed[worker_idx][sub_idx] = True

    return batch_obs, env_idxs, should_save

  def _update_experiences(self, output) -> List[Dict]:
    env_idx = output["env_idx"]
    if env_idx not in self.shared_dict["experiences"]:
      self.shared_dict["experiences"][env_idx] = []
    
    env_exps = self.shared_dict["experiences"][env_idx]
    for i, exp in enumerate(env_exps):
      exp["return"] += (self.gamma**i) * output["reward"]
    
    to_store = []
    if output["episode_done"]:
      while len(env_exps):
        exp = env_exps.pop()
        exp.update({
          "nth_state": output["observation"],
          "done": True,
        })
        to_store.append(exp)
    else:
      if len(env_exps) >= self.n_steps:
        exp = env_exps.pop()
        exp.update({
          "nth_state": output["observation"],
          "done": False,
        })
        to_store.append(exp)

      env_exps.insert(0, {
        "state": output["observation"],
        "return": 0.,
      })

    return to_store

  def _wait_for_actors(self):
    print("Waiting for actors to be initialized...")
    while self.process_state["running"] and not self.process_state["actors_initialized"]:
      time.sleep(ACTOR_WAIT_TIMEOUT)
    print("Actors intialized. Sending observations...")
