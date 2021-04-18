from queue import Queue
import torch
from typing import Any, Dict, List, Type

class ActorWorker:
  def __init__(
    self,
    actor_class: Type,
    input_queue: Queue,
    output_queue: Queue,
    update_queue: Queue,
    actor_config: Dict = {},
    index: int = 0,
    **kwargs,
  ):
    self.index = index

    self.input_queue = input_queue
    self.output_queue = output_queue
    self.update_queue = update_queue

    self.actor = actor_class(**actor_config)

    self.run()
  
  def run(self):
    update = self.update_queue.get()
    self._update_actor(update)

    while True:
      task = self.input_queue.get()
      obs = task["observation"]

      action = self._choose_action(obs)
      self.output_queue.put({
        "action": action,
        "worker_idx": task["worker_idx"],
        "env_idx": task["env_idx"],
      })

      if self.update_queue.qsize() > 0:
        update = self.update_queue.get()
        self._update_actor(update)

  def _update_actor(self, update: Any):
    self.actor.update(update)

  def _choose_action(self, obs: torch.Tensor):
    action = self.actor.choose_action(obs)
    return action
  