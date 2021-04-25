from multiprocessing import Process
from multiprocessing.queues import Queue
import numpy as np
import signal
from typing import Any, Dict, List, Type

from asyreile.environments.base import BaseEnvironment


class EnvironmentWorker(Process):
  def __init__(
    self,
    input_queue: Queue,
    output_queue: Queue,
    seed_seq: np.random.SeedSequence,
    env_class: Type[BaseEnvironment],
    num_envs: int = 2,
    max_episodes: int = 1,
    env_config: Dict = {},
    index: int = 0,
    **kwargs,
  ):
    super().__init__()

    self.input_queue = input_queue
    self.output_queue = output_queue
    self.seed_seq = seed_seq

    self.env_class = env_class
    self.num_envs = num_envs
    self.max_episodes = max_episodes
    self.env_config = env_config

    self.index = index

  def setup(self):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    self.envs = [self.env_class(**self.env_config) for _ in range(self.num_envs)]
    self.env_seed_seqs = self.seed_seq.spawn(self.num_envs)
    for i, seed_seq in enumerate(self.env_seed_seqs):
      self.envs[i].seed(seed_seq.generate_state(1).item())

    self.episode_scores = [0. for _ in range(self.num_envs)]
    self.episode_exp = [
      {"worker_idx": self.index, "env_idx": i} 
      for i in range(self.num_envs)
    ]

    self.total_steps = 0
    self.total_episodes = 0

  def run(self):
    self.setup()

    for idx in range(self.num_envs):
      exp = self._reset_env(idx)
      self.output_queue.put(exp)

    while True:
      task = self.input_queue.get()
      idx = task["env_idx"]

      if self.episode_exp[idx]["episode_done"]:
        exp = self._reset_env(idx)
        self.output_queue.put(exp)
      else:
        action = task["action"]
        exp = self._step_env(idx, action)
        self.output_queue.put(exp)
      
      if self.total_episodes >= self.max_episodes: break

  def _reset_env(self, idx: int) -> Dict:
    obs = self.envs[idx].reset()

    self.episode_scores[idx] = 0

    exp = self.episode_exp[idx]
    exp.update({
      "observation": obs,
      "reward": 0.,
      "episode_step": 0,
      "episode_done": False,
      "task_done": False,
    })

    return exp

  def _step_env(self, idx: int, action: Any) -> Dict:
    obs, reward, done, info = self.envs[idx].step(action)

    self.episode_scores[idx] += reward

    exp = self.episode_exp[idx]
    exp.update({
      "observation": obs,
      "reward": reward,
      "episode_step": exp["episode_step"] + 1,
      "episode_done": done,
      "task_done": done and not info.get("TimeLimit.truncated", False),
    })

    if done: self.total_episodes += 1
    self.total_steps += 1

    return exp 
