from multiprocessing import Queue, Process
import numpy as np
import random
import torch
from typing import Any, Dict, List, Type

from asyreile.actors.base import BaseActor

class ActorWorker:
  def __init__(
    self,
    input_queue: Queue,
    output_queue: Queue,
    sync_queue: Queue,
    update_queue: Queue,
    seed_seq: np.random.SeedSequence,
    actor_class: Type[BaseActor],
    actor_config: Dict = {},
    index: int = 0,
    **kwargs,
  ):
    super().__init__()

    self.input_queue = input_queue
    self.output_queue = output_queue
    self.sync_queue = sync_queue
    self.update_queue = update_queue
    self.seed_seq = seed_seq

    self.actor_class = actor_class
    self.actor_config = actor_config

    self.index = index

  def setup(self):
    self._set_worker_rng()
    self.actor = actor_class(**actor_config)

  def run(self):    
    self.setup()

    self._get_updates(wait=True)
    while True:
      task = self.input_queue.get()
      obs = task["observation"]

      action = self._choose_action(obs)
      self.output_queue.put({
        "action": action,
        "worker_idx": task["worker_idx"],
        "env_idx": task["env_idx"],
      })

      self._get_updates()

  def _set_worker_rng(self):
    np.random.seed(self.seed_seq.generate_state(4))

    torch_seed_seq, py_seed_seq = self.seed_seq.spawn(3)
    torch.manual_seed(torch_seed_seq.generate_state(1, dtype=np.uint64).item())

    py_seed = (py_seed_seq.generate_state(2, dtype=np.uint64).astype(object) * [1 << 64, 1]).sum()
    random.seed(py_seed)

  def _get_updates(self, wait=False):
    if not wait or self.update_queue.qsize() > 0:
      actor_params = self.update_queue.get()
      self._update_actor(actor_params)

    if not wait or self.sync_queue.qsize() > 0:
      network_params = self.sync_queue.get()
      self._sync_actor_networks(network_params)

  def _choose_action(self, obs: Any, greedy: bool = False) -> Any:
    action = self.actor.choose_action(obs, greedy=greedy)
    return action

  def _sync_actor_networks(self, network_params: Any):
    self.actor.sync_networks(network_params)
  
  def _update_actor(self, actor_params: Dict):
    self.actor.update(**actor_params)