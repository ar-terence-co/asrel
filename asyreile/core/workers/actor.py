from collections import OrderedDict
import torch.multiprocessing as mp
from multiprocessing.queues import Queue
import numpy as np
import random
import torch
from typing import Any, Dict, List, Type

from asyreile.actors.base import BaseActor
import asyreile.core.workers.events as events
from asyreile.core.utils import get_tensor_from_event

class ActorWorker(mp.Process):
  def __init__(
    self,
    input_queue: Queue,
    output_queue: Queue,
    seed_seq: np.random.SeedSequence,
    actor_class: Type[BaseActor],
    actor_config: Dict = {},
    index: int = 0,
    **kwargs,
  ):
    super().__init__()

    self.input_queue = input_queue
    self.output_queue = output_queue
    self.seed_seq = seed_seq

    self.actor_class = actor_class
    self.actor_config = actor_config

    self.index = index

  def setup(self):
    self._set_worker_rng()
    self.actor = self.actor_class(**self.actor_config)

  def run(self):    
    self.setup()

    while True:
      task = self.input_queue.get()
      task_type = task["type"]

      if task_type == events.ACTOR_CHOOSE_ACTION_TASK:
        obs = get_tensor_from_event(task, "observation")

        greedy = task.get("greedy", False)

        action = self._choose_action(obs, greedy=greedy)

        self.output_queue.put({
          "type": events.RETURNED_ACTION_EVENT,
          "action": action,
          "env_idx": task["env_idx"],
          "actor_idx": self.index,
        })

      elif task_type == events.ACTOR_SYNC_NETWORKS_TASK:
        state_dicts = task["state_dicts"]
        self._sync_actor_networks(state_dicts)

      elif task_type == events.ACTOR_UPDATE_PARAMS_TASK:
        self._update_actor(task)

  def _set_worker_rng(self):
    np.random.seed(self.seed_seq.generate_state(4))

    torch_seed_seq, py_seed_seq = self.seed_seq.spawn(2)
    torch.manual_seed(torch_seed_seq.generate_state(1, dtype=np.uint64).item())

    py_seed = (py_seed_seq.generate_state(2, dtype=np.uint64).astype(object) * [1 << 64, 1]).sum()
    random.seed(py_seed)

  def _choose_action(self, obs: torch.Tensor, greedy: bool = False) -> torch.Tensor:
    action = self.actor.choose_action(obs, greedy=greedy)
    return action

  def _sync_actor_networks(self, state_dicts: Dict[str, OrderedDict]):
    self.actor.sync_networks(state_dicts)
  
  def _update_actor(self, actor_params: Dict):
    self.actor.update(**actor_params)
