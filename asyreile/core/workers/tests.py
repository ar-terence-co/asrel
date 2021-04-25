import json
from multiprocessing import Process, get_context
from multiprocessing.queues import Queue, Empty
import numpy as np
import time
from threading import Thread
from typing import Callable, Dict, List

def test_environment_worker(config: Dict, seeds: Dict[str, np.random.SeedSequence]):
  """
  Function to run an isolated env worker and manually test it out in the cli.
  """
  from asyreile.core.utils import get_env_args_from_config
  from asyreile.core.workers.environment import EnvironmentWorker

  env_args = get_env_args_from_config(config["environment"])

  print(f"Testing Environment Worker with args: {env_args}")

  num_workers = env_args.get("num_workers", 1)
  num_envs = env_args.get("num_envs", 2)

  print("Creating workers...")

  ctx = get_context()
  env_input_queues = [Queue(maxsize=num_envs, ctx=ctx) for _ in range(num_workers)]
  env_output_queue = Queue(maxsize=num_workers * num_envs, ctx=ctx)
  env_worker_seed_seqs = seeds["environment"].spawn(num_workers)

  env_workers = [
    EnvironmentWorker(
      input_queue=env_input_queues[idx],
      output_queue=env_output_queue,
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
      env_out = env_output_queue.get()
      print(f"worker {env_out['worker_idx']}:")
      print({**env_out, "observation": f"... [{env_out['observation'].shape}]"})
      while env_output_queue.qsize() > 0:
        env_out = env_output_queue.get()
        print(f"worker {env_out['worker_idx']}:")
        print({**env_out, "observation": f"... [{env_out['observation'].shape}]"})
      
      print("in: ")
      worker_idx = int(input(" worker: ",))
      env_idx = int(input(" env:    "))
      action = int(input(" action: "))
      env_input_queues[worker_idx].put({
        "worker_idx": worker_idx,
        "env_idx": env_idx,
        "action": action,
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
