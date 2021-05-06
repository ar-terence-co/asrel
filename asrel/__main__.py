import numpy as np
import torch.multiprocessing as mp
import pathlib
import sys
import time
import torch
from threading import Thread
from typing import Dict, List

from asrel.core.utils import (
  get_args, 
  get_config, 
  get_seed_sequences, 
  get_registry_args_from_config,
  get_orchestrator_from_config,
  get_env_args_from_config, 
  get_actor_args_from_config,
  get_store_args_from_config,
  get_learner_args_from_config,
  get_spaces_from_env_args,
  DEFAULT_CHECKPOINT_DIR,
)

def main(config: Dict, seeds: List[np.random.SeedSequence]):
  from asrel.core.orchestrator import Orchestrator
  from asrel.core.registry import WorkerRegistry
  from asrel.core.workers.environment import EnvironmentWorker
  from asrel.core.workers.actor import ActorWorker
  from asrel.core.workers.store import ExperienceStoreWorker
  from asrel.core.workers.learner import LearnerWorker

  mp.set_start_method("spawn")

  env_args = get_env_args_from_config(config["environment"])
  observation_space, action_space = get_spaces_from_env_args(env_args)
  
  global_config = {
    **config.get("global", {}),
    "input_space": observation_space,
    "output_space": action_space,
  }

  print("Creating registry...")
  registry_args = get_registry_args_from_config(config["registry"])
  registry = WorkerRegistry(**registry_args)

  print("Creating env workers...")
  env_workers = []
  env_worker_args = env_args
  num_workers = env_worker_args["num_workers"]
  num_envs = env_worker_args["num_envs"]
  env_worker_seed_seqs = seeds["environment"].spawn(num_workers)
  for seed_seq in env_worker_seed_seqs:
    idx, input_queue, output_queue = registry.register(
      "environment", 
      env_worker_args, 
      input_maxsize=num_envs,
      output_maxsize=num_envs,
    )
    worker = EnvironmentWorker(
      input_queue=input_queue,
      output_queue=output_queue,
      seed_seq=seed_seq,
      global_config=global_config,
      index=idx,
      **env_worker_args,
    )
    env_workers.append(worker)

  print("Creating actor workers...")
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
      global_config=global_config,
      index=idx,
      **actor_worker_args,
    )
    actor_workers.append(worker)
  
  print("Creating experience store worker...")
  store_worker_args = get_store_args_from_config(config["store"])
  buffer_size = store_worker_args["buffer_size"]
  idx, input_queue, output_queue = registry.register(
    "store", 
    store_worker_args, 
    output_maxsize=buffer_size
  )
  store_worker = ExperienceStoreWorker(
    input_queue=input_queue,
    output_queue=output_queue,
    seed_seq=seeds["store"],
    global_config=global_config,
    index=idx,
    **store_worker_args,
  )

  print("Creating learner worker...")
  learner_worker_args = get_learner_args_from_config(config["learner"])
  idx, input_queue, output_queue = registry.register(
    "learner",
    learner_worker_args,
    input_maxsize=buffer_size,
  )
  learner_worker = LearnerWorker(
    input_queue=input_queue,
    output_queue=output_queue,
    seed_seq=seeds["learner"],
    global_config=global_config,
    index=idx,
    **learner_worker_args,
  )

  print("Creating orchestrator...")
  orch = Orchestrator(
    registry=registry,
    seed_seq=seeds["orchestrator"],
    pipeline_config=config["pipelines"],
    global_config=global_config,
  )

  all_workers = [
    orch,
    *env_workers,
    *actor_workers,
    store_worker,
    learner_worker,
  ]

  print("Starting workers...")
  for worker in all_workers: worker.start()

  try:
    orch.join()
    close_workers(all_workers)
  except (KeyboardInterrupt, Exception) as e:
    print()
    print(e)
    orch.terminate()
    close_workers(all_workers)
  
  
def close_workers(workers: mp.Process):
  for worker in workers: worker.join()
  for worker in workers: worker.close()

if __name__ == "__main__":
  args = get_args()
  config = get_config(args.config)
  seeds = get_seed_sequences(config)

  if args.test_env_worker:
    from asrel.core.workers.tests import test_environment_worker
    test_environment_worker(config, seeds)
    sys.exit()

  if args.test_actor_worker:
    from asrel.core.workers.tests import test_actor_worker
    test_actor_worker(config, seeds)
    sys.exit()

  main(config, seeds)
