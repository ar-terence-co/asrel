import numpy as np
from queue import Queue
import sys
import torch
from threading import Thread
from typing import Dict, List

from asyreile.core.utils import (
  get_args, 
  get_config, 
  get_seed_sequences, 
  get_env_args_from_config, 
  get_actor_args_from_config,
  get_spaces_from_env_args,
)

def main(config: Dict, seeds: List[np.random.SeedSequence]):
  from asyreile.core.orchestrators.simple import SimpleOrchestrator
  from asyreile.core.workers.environment import EnvironmentWorker
  from asyreile.core.workers.actor import ActorWorker

  print("Creating orchestrator...")
  orch = SimpleOrchestrator(seeds["orchestrator"], split_env_processing = 2)

  env_args = get_env_args_from_config(config["environment"])
  observation_space, action_space = get_spaces_from_env_args(env_args)

  print("Creating env workers...")
  env_workers = []
  env_worker_args = env_args
  num_workers = env_worker_args["num_workers"]
  env_worker_seed_seqs = seeds["environment"].spawn(num_workers)
  for seed_seq in env_worker_seed_seqs:
    idx, input_queue, output_queue = orch.register_env_worker(env_worker_args)
    worker = EnvironmentWorker(
      input_queue=input_queue,
      output_queue=output_queue,
      seed_seq=seed_seq,
      index=idx,
      **env_worker_args,
    )
    env_workers.append(worker)

  print("Creating action workers...")
  actor_workers = []
  actor_worker_args = get_actor_args_from_config(config["actor"])
  num_workers = actor_worker_args["num_workers"]
  actor_worker_seed_seqs = seeds["actor"].spawn(num_workers)
  for seed_seq in actor_worker_seed_seqs:
    idx, input_queue, output_queue = orch.register_actor_worker(actor_worker_args)
    worker = ActorWorker(
      input_queue=input_queue,
      output_queue=output_queue,
      seed_seq=seed_seq,
      index=idx,
      **actor_worker_args,
    )
    actor_workers.append(worker)

if __name__ == "__main__":
  args = get_args()
  config = get_config(args.config)
  seeds = get_seed_sequences(config)

  # setup()

  if args.test_env_worker:
    from asyreile.core.workers.tests import test_environment_worker
    test_environment_worker(config, seeds)
    sys.exit()

  if args.test_actor_worker:
    from asyreile.core.workers.tests import test_actor_worker
    test_actor_worker(config, seeds)
    sys.exit()

  main(config, seeds)

  
