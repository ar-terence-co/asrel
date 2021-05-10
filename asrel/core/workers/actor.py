from collections import OrderedDict
from gym import Space
import torch.multiprocessing as mp
from multiprocessing.queues import Queue
import numpy as np
import signal
import torch
from typing import Any, Dict, List, Type

from asrel.actors.base import BaseActor
import asrel.core.workers.events as events
from asrel.core.utils import take_tensor_from_dict, take_tensors_from_state_dicts, set_worker_rng, validate_subclass

class ActorWorker(mp.Process):
  def __init__(
    self,
    input_queue: Queue,
    output_queue: Queue,
    seed_seq: np.random.SeedSequence,
    actor_class: Type[BaseActor],
    actor_config: Dict = {},
    global_config: Dict = {},
    index: int = 0,
    **kwargs,
  ):
    super().__init__()

    self.input_queue = input_queue
    self.output_queue = output_queue
    self.seed_seq = seed_seq

    validate_subclass(actor_class, BaseActor)
    self.actor_class = actor_class
    self.actor_config = actor_config

    self.global_config = global_config
    self.index = index

  def setup(self):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    print(f"Started {mp.current_process().name}.")

    set_worker_rng(self.seed_seq)
    self.actor = self.actor_class(
      **self.actor_config, 
      global_config=self.global_config,
    )

  def run(self):    
    self.setup()

    while True:
      task = self.input_queue.get()
      task_type = task["type"]

      if task_type == events.ACTOR_CHOOSE_ACTION_TASK:
        obs = take_tensor_from_dict(task, "observation")

        greedy = task.get("greedy", False)

        action = self._choose_action(obs, greedy=greedy)

        self.output_queue.put({
          "type": events.RETURNED_ACTION_EVENT,
          "action": action,
          "env_idx": task["env_idx"],
          "actor_idx": self.index,
        })

      elif task_type == events.ACTOR_SYNC_NETWORKS_TASK:
        state_dicts = take_tensors_from_state_dicts(task["state_dicts"])
        self._sync_actor_networks(state_dicts)

      elif task_type == events.ACTOR_UPDATE_PARAMS_TASK:
        self._update_actor(task)

      elif task_type == events.WORKER_TERMINATE_TASK:
        break

    self.cleanup()

  def cleanup(self):
    print(f"Terminated {mp.current_process().name}.")

  def _choose_action(self, obs: torch.Tensor, greedy: bool = False) -> torch.Tensor:
    action = self.actor.choose_action(obs, greedy=greedy)
    return action

  def _sync_actor_networks(self, state_dicts: Dict[str, OrderedDict]):
    self.actor.sync_networks(state_dicts)
  
  def _update_actor(self, actor_params: Dict):
    self.actor.update(**actor_params)
