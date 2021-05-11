from multiprocessing.queues import Empty
import torch
from typing import Dict

import asrel.core.workers.events as events
from asrel.pipelines.base import BasePipeline

class StandardDatasetPipeline(BasePipeline):
  """
  Standard dataset pipeline 
  """
  def __init__(
    self,
    **kwargs,
  ):
    super().__init__(**kwargs)

    self.store_output_queue = self.registry.output_queues["store"][0]
    self.learner_input_queue = self.registry.input_queues["learner"][0]

    config = self.registry.configs["learner"][0]
    self.learner_device = torch.device(
      config.get("learner_config", {}).get("device", "cuda")
    )

  def run(self):
    while self.process_state["running"]:
      output = self._get_output()
      if not output: continue

      batch_tensor = self._process_output(output)

      task = {
        "type": events.LEARNER_TRAIN_TASK,
        "data": batch_tensor,
      }
      self.send_task(self.learner_input_queue, task)

  def _get_output(self) -> Dict:
    try:
      output = self.store_output_queue.get(block=True, timeout=self.queue_timeout)

      if output["type"] != events.RETURNED_BATCH_DATA_EVENT:
        return None # Note this skips other returned events from the actor

      return output
    except Empty:
      return None

  def _process_output(self, output: Dict) -> Dict[str, torch.Tensor]:
    batch_tensor = {
      k: torch.tensor(v, device=self.learner_device)
      for k, v in output["data"].items()
    }
    return batch_tensor
