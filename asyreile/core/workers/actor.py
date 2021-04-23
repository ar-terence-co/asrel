from queue import Queue
from typing import Any, Dict, List, Type

class ActorWorker:
  def __init__(
    self,
    actor_class: Type,
    input_queue: Queue,
    output_queue: Queue,
    sync_queue: Queue,
    update_queue: Queue,
    actor_config: Dict = {},
    index: int = 0,
    **kwargs,
  ):
    self.index = index

    self.input_queue = input_queue
    self.output_queue = output_queue
    self.sync_queue = sync_queue
    self.update_queue = update_queue

    self.actor = actor_class(**actor_config)

    self.run()
  
  def run(self):    
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