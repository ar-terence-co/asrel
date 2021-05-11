from collections import OrderedDict
from multiprocessing.queues import Empty
import numpy as np
from typing import Dict, List, Tuple

from asrel.core.utils import take_tensors_from_state_dicts
import asrel.core.workers.events as events
from asrel.pipelines.base import BasePipeline

class StandardPolicyPipeline(BasePipeline):
  """
  Standard observation pipeline 
  """
  def __init__(
    self,
    **kwargs,
  ):
    super().__init__(**kwargs)
    
    self.leaner_output_queue = self.registry.output_queues["learner"][0]
    self.actor_input_queues = self.registry.input_queues["actor"]

    self.actor_q_count = len(self.actor_input_queues)

  def run(self):
    actors_intialized = self.process_state["actors_initialized"]

    while self.process_state["running"]:
      output = self._get_output()
      if not output: continue

      if output["type"] == events.RETURNED_NETWORK_UPDATE_EVENT:
        state_dicts = take_tensors_from_state_dicts(output["state_dicts"])
        task = {
          "type": events.ACTOR_SYNC_NETWORKS_TASK,
          "state_dicts": state_dicts,
        }  
      elif output["type"] == events.RETURNED_ACTOR_UPDATE_EVENT:
        task = dict(
          output,
          type=events.ACTOR_UPDATE_PARAMS_TASK,
        )
      else:
        continue

      for actor_input_queue in self.actor_input_queues:
        self.send_task(actor_input_queue, task)
      
      if not actors_intialized and task["type"] == events.ACTOR_SYNC_NETWORKS_TASK:
        actors_intialized = True
        self.process_state["actors_initialized"] = True

  def _get_output(self) -> Dict:
    try:
      output = self.leaner_output_queue.get(block=True, timeout=self.queue_timeout)
      return output
    except Empty:
      return None
