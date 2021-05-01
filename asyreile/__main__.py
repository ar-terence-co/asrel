import numpy as np
import torch.multiprocessing as mp
import sys
import torch
from threading import Thread
from typing import Dict, List

from asyreile.core.utils import (
  get_args, 
  get_config, 
  get_seed_sequences, 
  get_registry_args_from_config,
  get_orchestrator_from_config,
  get_env_args_from_config, 
  get_actor_args_from_config,
  get_spaces_from_env_args,
)

def main(config: Dict, seeds: List[np.random.SeedSequence]):
  from asyreile.core.orchestrator import Orchestrator
  from asyreile.core.registry import WorkerRegistry
  from asyreile.core.workers.environment import EnvironmentWorker
  from asyreile.core.workers.actor import ActorWorker

  env_args = get_env_args_from_config(config["environment"])
  observation_space, action_space = get_spaces_from_env_args(env_args)

  print("Creating registry...")
  registry_args = get_registry_args_from_config(config["registry"])
  registry = WorkerRegistry(**registry_args)

  # print("Creating orchestrator...")
  # orch = get_orchestrator_from_config(config["orchestrator"], seeds["orchestrator"])

  print("Creating env workers...")
  env_workers = []
  env_worker_args = env_args
  num_workers = env_worker_args["num_workers"]
  num_envs = env_worker_args["num_envs"]
  env_worker_seed_seqs = seeds["environment"].spawn(num_workers)
  for seed_seq in env_worker_seed_seqs:
    idx, input_queue, output_queue = registry.register("environment", env_worker_args, maxsize=num_envs)
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
    idx, input_queue, output_queue = registry.register("actor", actor_worker_args)
    worker = ActorWorker(
      input_queue=input_queue,
      output_queue=output_queue,
      seed_seq=seed_seq,
      input_space=observation_space,
      output_space=action_space,
      index=idx,
      **actor_worker_args,
    )
    actor_workers.append(worker)

  print("Creating orchestrator...")
  orch = Orchestrator(
    registry=registry,
    seed_seq=seeds["orchestrator"],
    pipeline_config=config["pipelines"],
    max_episodes=config.get("max_episodes", 100),
  )

  all_workers = [
    orch,
    *env_workers,
    *actor_workers,
  ]

  print("Starting workers...")
  for worker in all_workers: worker.start()

  try:
    orch.join()
    terminate_workers(all_workers)
  except (KeyboardInterrupt, Exception) as e:
    print()
    print(e)
    terminate_workers(all_workers)
  
  
def terminate_workers(workers: mp.Process):
  print("Terminating workers...")
  for worker in workers: worker.terminate()
  for worker in workers: worker.join()
  for worker in workers: worker.close()
  print("Terminated successfully.")


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
