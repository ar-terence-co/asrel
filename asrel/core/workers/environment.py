import torch.multiprocessing as mp
from multiprocessing.queues import Queue
import numpy as np
import signal
from typing import Any, Dict, List, Type

from asrel.core.utils import validate_subclass
import asrel.core.workers.events as events
from asrel.environments.base import BaseEnvironment


class EnvironmentWorker(mp.Process):
  def __init__(
    self,
    input_queue: Queue,
    output_queue: Queue,
    seed_seq: np.random.SeedSequence,
    env_class: Type[BaseEnvironment],
    num_envs: int = 2,
    env_config: Dict = {},
    global_config: Dict = {},
    index: int = 0,
    **kwargs,
  ):
    super().__init__()

    self.input_queue = input_queue
    self.output_queue = output_queue
    self.seed_seq = seed_seq

    validate_subclass(env_class, BaseEnvironment)
    self.env_class = env_class
    self.num_envs = num_envs
    self.env_config = env_config

    self.global_config = global_config
    self.index = index

  def setup(self):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    print(f"Started {mp.current_process().name}.")

    self.envs = [self.env_class(**self.env_config) for _ in range(self.num_envs)]
    self.env_seed_seqs = self.seed_seq.spawn(self.num_envs)
    for i, seed_seq in enumerate(self.env_seed_seqs):
      self.envs[i].seed(seed_seq.generate_state(1).item())

    self.episode_scores = [0. for _ in range(self.num_envs)]
    self.episode_steps = [0 for _ in range(self.num_envs)]
    self.episode_dones = [True for _ in range(self.num_envs)]
    
    self.total_steps = 0

  def run(self):
    self.setup()

    for idx in range(self.num_envs):
      obs = self._reset_env(idx)
      self.output_queue.put({
        "type": events.RETURNED_OBSERVATION_EVENT,
        "observation": obs,
        "reward": 0.,
        "score": 0.,
        "episode_step": 0,
        "episode_done": False,
        "task_done": False,
        "env_idx": (self.index, idx),
      })

    while True:
      task = self.input_queue.get()
      task_type = task["type"]

      if task_type == events.ENV_INTERACT_TASK:
        _, idx = task["env_idx"]

        if self.episode_dones[idx]:
          obs = self._reset_env(idx)
          self.output_queue.put({
            "type": events.RETURNED_OBSERVATION_EVENT,
            "observation": obs,
            "reward": 0.,
            "score": 0.,
            "episode_step": 0,
            "episode_done": False,
            "task_done": False,
            "env_idx": (self.index, idx),
          })
        else:
          action = task["action"]
          obs, reward, done, info = self._step_env(idx, action)
          self.output_queue.put({
            "type": events.RETURNED_OBSERVATION_EVENT,
            "observation": obs,
            "reward": reward,
            "score": self.episode_scores[idx],
            "episode_step": self.episode_steps[idx],
            "episode_done": done,
            "task_done": done and not info.get("TimeLimit.truncated", False),
            "env_idx": (self.index, idx),
          })

      elif task_type == events.WORKER_TERMINATE_TASK:
        break

    self.cleanup()
      
  def cleanup(self):
    print(f"Terminated {mp.current_process().name}.")

  def _reset_env(self, idx: int) -> Dict:
    obs = self.envs[idx].reset()

    self.episode_scores[idx] = 0
    self.episode_steps[idx] = 0
    self.episode_dones[idx] = False

    return obs

  def _step_env(self, idx: int, action: Any) -> Dict:
    obs, reward, done, info = self.envs[idx].step(action)

    self.episode_scores[idx] += reward
    self.episode_steps[idx] += 1
    self.episode_dones[idx] = done

    self.total_steps += 1

    return obs, reward, done, info
