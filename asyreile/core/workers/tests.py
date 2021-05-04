import json
import torch.multiprocessing as mp
import numpy as np
import time
import torch
from threading import Thread
from typing import Callable, Dict, List

def test_environment_worker(config: Dict, seeds: Dict[str, np.random.SeedSequence]):
  """
  Function to run an isolated env workers and manually test it out in the cli.
  """
  from asyreile.core.utils import get_env_args_from_config
  from asyreile.core.workers.environment import EnvironmentWorker
  import asyreile.core.workers.events as events

  env_args = get_env_args_from_config(config["environment"])

  print(f"Testing Environment Worker with args: {env_args}")

  num_workers = env_args.get("num_workers", 1)
  num_envs = env_args.get("num_envs", 2)

  print("Creating workers...")

  env_input_queues = [mp.Queue(maxsize=num_envs) for _ in range(num_workers)]
  env_shared_output_queue = mp.Queue(maxsize=num_workers * num_envs)
  env_worker_seed_seqs = seeds["environment"].spawn(num_workers)

  env_workers = [
    EnvironmentWorker(
      input_queue=env_input_queues[idx],
      output_queue=env_shared_output_queue,
      seed_seq=env_worker_seed_seqs[idx],
      index=idx,
      **env_args,
    )
    for idx in range(num_workers)
  ]
  for worker in env_workers:
    worker.start()

  print(f"Started workers...")

  try:
    while True:
      time.sleep(2)
      out = env_shared_output_queue.get()
      worker_idx, _ = out["env_idx"]
      print(f"worker {worker_idx}:")
      print({**out, "observation": f"... [{out['observation'].shape}]"})
      while env_shared_output_queue.qsize() > 0:
        out = env_shared_output_queue.get()
        worker_idx, _ = out["env_idx"]
        print(f"worker {worker_idx}:")
        print({**out, "observation": f"... [{out['observation'].shape}]"})
      
      print("in: ")
      task = int(input("0 - Interact: "))
      if task == 0:
        worker_idx = int(input(" worker: ",))
        sub_idx = int(input(" sub:    "))
        action = int(input(" action: "))
        env_input_queues[worker_idx].put({
          "type": events.ENV_INTERACT_TASK,
          "action": action,
          "env_idx": (worker_idx, sub_idx),
        })
  except (KeyboardInterrupt, Exception) as e:
    print()
    print("Terminating workers...")
    for worker in env_workers:
      worker.terminate()
    print(e)
  else:
    print("Closing worker...")
    for worker in env_workers:
      worker.close()

  for worker in env_workers:
    worker.join()

def test_actor_worker(config: Dict, seeds: List[np.random.SeedSequence]):
  """
  Function to run an isolated actor workers and manually test it out in the cli.
  """
  from gym.spaces import Box, Discrete
  from asyreile.core.utils import get_actor_args_from_config, get_tensor_from_event
  from asyreile.core.workers.actor import ActorWorker
  import asyreile.core.workers.events as events

  actor_args = get_actor_args_from_config(config["actor"])

  print(f"Testing Actor Worker with args: {actor_args}")

  num_workers = actor_args.get("num_workers", 1)
  input_queue_len = 8
  input_space = Box(-10, 10, (6, ), np.float32)

  input_space.seed(0)
  output_space = Discrete(3)

  print("Creating workers...")

  actor_input_queues = [mp.Queue(maxsize=input_queue_len) for _ in range(num_workers)]
  actor_shared_output_queue = mp.Queue(maxsize=num_workers*input_queue_len)
  actor_worker_seed_seqs = seeds["actor"].spawn(num_workers)

  actor_workers = [
    ActorWorker(
      input_queue=actor_input_queues[idx],
      output_queue=actor_shared_output_queue,
      seed_seq=actor_worker_seed_seqs[idx],
      input_space=input_space,
      output_space=output_space,
      index=idx,
      **actor_args,
    )
    for idx in range(num_workers)
  ]

  for worker in actor_workers:
    worker.start()

  try:
    while True:
      task = int(input("0 - Choose Action, 1 - Sync Networks, 2 - Update Params: "))
      if task == 0:
        worker_idx = int(input(" worker: ",))

        num_obs = int(input(" # of obs: "))
        obs = torch.tensor([input_space.sample() for _ in range(num_obs)]).cuda()
        print(f" obs:\n{obs}")
        env_worker_idx = int(input(" env worker:   "))
        env_sub_idx = int(input("     subenv:   "))
        greedy = input(" greedy (y/n): ").lower() == "y"
        actor_input_queues[worker_idx].put({
          "type": events.ACTOR_CHOOSE_ACTION_TASK,
          "observation": obs,
          "greedy": greedy,
          "env_idx": (env_worker_idx, env_sub_idx),
        })
        out = actor_shared_output_queue.get()
        out_action = get_tensor_from_event(out, "action")
        print(f"worker {worker_idx}:")
        print({**out, "action": out_action})

      elif task == 1:
        state_dicts = json.loads(input("State Dictionaries: "))
        for q in actor_input_queues:
          q.put({
            "type": events.ACTOR_SYNC_NETWORKS_TASK,
            "state_dicts": state_dicts,
          })
      elif task == 2:
        params = json.loads(input("Params: "))
        for q in actor_input_queues:
          q.put({
            "type": events.ACTOR_UPDATE_PARAMS_TASK,
            **params,
          })

  except (KeyboardInterrupt, Exception) as e:
    print()
    print("Terminating workers...")
    for worker in actor_workers:
      worker.terminate()
    print(e)
  else:
    print("Closing worker...")
    for worker in actor_workers:
      worker.close()

  for worker in actor_workers:
    worker.join()
