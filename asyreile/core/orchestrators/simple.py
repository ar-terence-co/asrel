import numpy as np
from threading import Thread
import torch.multiprocessing as mp
from multiprocessing.queues import Queue, Empty
import signal
import timeit
import torch
from typing import Dict, List, Tuple

from asyreile.core.utils import set_worker_rng, get_tensor_from_event
import asyreile.core.workers.events as events

class SimpleOrchestrator(mp.Process):
  def __init__(
    self,
    seed_seq: np.random.SeedSequence,
    max_episodes: int = 100,
    queue_timeout: int = 60,
    **kwargs
  ):
    super().__init__()

    self.seed_seq = seed_seq
    self.max_episodes = max_episodes
    self.queue_timeout = queue_timeout
    
    self.has_shared_env_queue = kwargs.get("has_shared_env_queue", False)
    if self.has_shared_env_queue:
      self.shared_env_queue_pull_count = kwargs.get("shared_env_queue_pull_count", 1)
    else:
      self.split_env_processing = kwargs.get("split_env_processing", 2)

    self.env_input_queues = []
    self.env_output_queues = []
    self.env_counts = []

    self.actor_input_queues = []
    self.actor_output_queues = []
    self.actor_device_names = []

  def register_env_worker(self, config: Dict) -> Tuple[int, Queue, Queue]:
    num_envs = config["num_envs"]

    input_queue = mp.Queue(maxsize=num_envs)
    self.env_input_queues.append(input_queue)

    if self.has_shared_env_queue:
      if len(self.env_output_queues) == 0:
        self.env_output_queues.append(mp.Queue(maxsize=0))
      output_queue = self.env_output_queues[0]
    else:
      output_queue = mp.Queue(maxsize=num_envs)
      self.env_output_queues.append(output_queue)

    idx = len(self.env_input_queues) - 1
    self.env_counts.append(num_envs)
    return idx, input_queue, output_queue

  def register_actor_worker(self, config: Dict) -> Tuple[int, Queue, Queue]:
    device_name = config.get("actor_config", {}).get("device", "cuda")
    self.actor_device_names.append(device_name)

    input_queue = mp.Queue(maxsize=0)
    self.actor_input_queues.append(input_queue)

    if len(self.actor_output_queues) == 0:
      self.actor_output_queues.append(mp.Queue(maxsize=0))
    output_queue = self.actor_output_queues[0]

    idx = len(self.actor_input_queues) - 1
    return idx, input_queue, output_queue

  def setup(self):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    set_worker_rng(self.seed_seq)

    if self.has_shared_env_queue:
      self.env_pulls = [self.shared_env_queue_pull_count]
    else:
      # NOTE: this expects the number of envs to be divisible by split_env_processing
      self.env_pulls = [c // self.split_env_processing for c in self.env_counts]

    self.actor_devices = [torch.device(device_name) for device_name in self.actor_device_names]

    self.env_stopped = [[False for _ in range(num_envs)] for num_envs in self.env_counts]
    self.total_episodes = 0
    self.total_steps = 0

    self.running = True

  def run(self):
    start_time = timeit.default_timer()

    self.setup()
    
    observation_pipeline = Thread(target=self._observation_pipeline)
    action_pipeline = Thread(target=self._action_pipeline)

    observation_pipeline.start()
    action_pipeline.start()

    observation_pipeline.join()
    action_pipeline.join()

    print(f"{self.total_episodes} episodes finished. {self.total_steps} steps ran. Orchestration ended.")

    end_time = timeit.default_timer()
    duration = end_time - start_time
    print(f"Duration: {duration} seconds")
    print(f"Average time steps per second: {duration / self.total_steps}")

  def _observation_pipeline(self):
    actor_idx = 0
    actor_count = len(self.actor_input_queues)

    while self.running:
      outputs = []
      pull_counts = [0 for _ in range(len(self.env_pulls))]

      # Pull from each queue the amount in self.env_pulls
      while True:
        has_pull = False
        for i in range(len(self.env_counts)):
          env_output_queue = self.env_output_queues[i]
          if all(self.env_stopped[i]): continue
          if pull_counts[i] >= self.env_pulls[i]: continue
          try:
            out = env_output_queue.get(block=True, timeout=self.queue_timeout)
          except Empty:
            continue
          outputs.append(out)
          has_pull = True
          pull_counts[i] += 1
        if not has_pull: break

      if not outputs:
        if self.total_episodes >= self.max_episodes:
          self.running = False
          break
        else:
          continue
      
      # Batch observations and create payload
      device = self.actor_devices[actor_idx]
      batch_obs = []
      env_idxs = []
      for out in outputs:
        if out["type"] != events.RETURNED_OBSERVATION_EVENT: 
          continue # Note this skips other returned events from the env
        batch_obs.append(out["observation"])
        env_idxs.append(out["env_idx"])
        self.total_steps += 1

        # Mark this env as stopped if over the total number of episodes
        if out["episode_done"]:
          self.total_episodes += 1
          if self.total_episodes >= self.max_episodes:
            worker_idx, sub_idx = out["env_idx"]
            self.env_stopped[worker_idx][sub_idx] = True

      task = {
        "type": events.ACTOR_CHOOSE_ACTION_TASK,
        "observation": torch.Tensor(batch_obs).to(device),
        "greedy": False,
        "env_idx": env_idxs,
        "actor_idx": actor_idx,
      }

      # Send to actor worker
      self.actor_input_queues[actor_idx].put(task)
      actor_idx = (actor_idx + 1) % actor_count

  def _action_pipeline(self):
    count = 0

    while self.running:
      # Pull from action output queue
      try:
        out = self.actor_output_queues[0].get(block=True, timeout=self.queue_timeout)
      except Empty:
        continue
      # print(count, out)
      if out["type"] != events.RETURNED_ACTION_EVENT: 
        continue # Note this skips other returned events from the actor
      actions_t = get_tensor_from_event(out, "action")
      actions = actions_t.cpu().numpy()
      env_idxs = out["env_idx"]

      # Split batch observation to correct workers
      for i, env_idx in enumerate(env_idxs):
        env_worker_idx, _ = env_idx
        action = actions[i].item()
        task = {
          "type": events.ENV_INTERACT_TASK,
          "action": action,
          "env_idx": env_idx,
        }

        # Send to env worker
        self.env_input_queues[env_worker_idx].put(task)
      
      count += 1
