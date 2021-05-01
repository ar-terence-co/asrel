from multiprocessing.queues import Empty
import numpy as np
from typing import Dict, List, Tuple

from asyreile.core.utils import get_tensor_from_event
import asyreile.core.workers.events as events
from asyreile.pipelines.base import BasePipeline

class StandardActionPipeline(BasePipeline):
  """
  Standard observation pipeline 
  """
  def __init__(
    self,
    **kwargs,
  ):
    super().__init__(**kwargs)
    
    self.actor_output_queues = self.registry.output_queues["actor"]
    self.env_input_queues = self.registry.input_queues["environment"]

    self.actor_q_count = len(self.actor_output_queues)
    self.env_q_count = len(self.env_input_queues)

  def run(self):
    actor_idx = 0

    while self.process_state["running"]:
      output = self._get_output(actor_idx)
      if not output: continue

      batch_actions, env_idxs = self._process_output(output)

      for i, env_idx in enumerate(env_idxs):
        task = {
          "type": events.ENV_INTERACT_TASK,
          "action": batch_actions[i].item(),
          "env_idx": env_idx,
        }

        env_worker_idx, _ = env_idx
        self.env_input_queues[env_worker_idx].put(task)
      
      actor_idx = (actor_idx + 1) % self.actor_q_count

  def _get_output(self, idx: int) -> Dict:
    try:
      output = self.actor_output_queues[idx].get(block=True, timeout=self.queue_timeout)

      if output["type"] != events.RETURNED_ACTION_EVENT:
        return None # Note this skips other returned events from the actor

      return output
    except Empty:
      return None

  def _process_output(self, output: Dict) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    batch_actions_t = get_tensor_from_event(output, "action")
    batch_actions = batch_actions_t.cpu().numpy()
    env_idxs = output["env_idx"]

    return batch_actions, env_idxs
