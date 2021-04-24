import json
from multiprocessing import Queue, Process
import numpy as np
import time
from typing import Dict, List


def test_environment_worker(config: Dict, seeds: Dict[str, np.random.SeedSequence]):
  """
  Function to run an isolated env worker and manually test it out in the cli.
  """
  from asyreile.core.utils import get_env_args_from_config
  from asyreile.core.workers.environment import EnvironmentWorker

  env_args = get_env_args_from_config(config["environment"])

  print(f"Testing Environment Worker with args: {env_args}")

  env_input_queue = Queue(maxsize=env_args["num_envs"])
  env_output_queue = Queue(maxsize=env_args["num_envs"])

  env_worker = EnvironmentWorker(
    input_queue=env_input_queue,
    output_queue=env_output_queue,
    seed_seq=seeds["environment"],
    **env_args,
  )
  env_worker.start()

  print(f"Started worker...")

  try:
    while True:
      time.sleep(1)
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
      env_input_queue.put({
        "worker_idx": worker_idx,
        "env_idx": env_idx,
        "action": action,
      })
  except (KeyboardInterrupt, Exception) as e:
    print()
    print("Terminating worker...")
    env_worker.terminate()
    print(e)
  else:
    print("Closing worker...")
    env_worker.close()

  env_worker.join()
