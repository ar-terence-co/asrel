import torch.multiprocessing as mp
from typing import Dict, List

from asrel.core.orchestrator import Orchestrator
from asrel.core.registry import WorkerRegistry
from asrel.core.utils import (
  get_config, 
  get_seed_sequences, 
  get_registry_args_from_config,
  get_env_args_from_config, 
  get_actor_args_from_config,
  get_store_args_from_config,
  get_learner_args_from_config,
  get_orchestrator_args_from_config,
  get_spaces_from_env_args,
)
from asrel.core.workers.actor import ActorWorker
from asrel.core.workers.environment import EnvironmentWorker
from asrel.core.workers.learner import LearnerWorker
from asrel.core.workers.store import ExperienceStoreWorker

class Runner:
  def __init__(
    self,
    config_file: str,
  ):
    self.config = get_config(config_file)
    self.seed_seqs = get_seed_sequences(self.config)

    mp.set_start_method("spawn")

    self.global_config = self._get_global_config()
    self.registry = self._create_registry()
    self.orchestrator = self._create_orchestrator()
    self.workers = self._create_workers()

  def start(self):
    print("Starting workers")
    
    self.orchestrator.start()
    for worker in self.workers: worker.start()

    try:
      self.orchestrator.join()
      self._close_workers()
    except(KeyboardInterrupt, Exception) as e:
      print()
      print(e)
      self.orchestrator.terminate()
      self._close_workers()

  def _get_global_config(self):
    env_args = get_env_args_from_config(self.config["environment"])
    observation_space, action_space = get_spaces_from_env_args(env_args)

    return {
      **self.config.get("global", {}),
      "input_space": observation_space,
      "output_space": action_space,
    }

  def _create_registry(self) -> WorkerRegistry:
    print("Creating registry...")
    registry_args = get_registry_args_from_config(self.config["registry"])
    return WorkerRegistry(**registry_args)

  def _create_orchestrator(self) -> Orchestrator:
    print("Creating orchestrator...")
    
    orchestrator_args = get_orchestrator_args_from_config(self.config["orchestrator"])
    
    orchestrator = Orchestrator(
      registry=self.registry,
      seed_seq=self.seed_seqs["orchestrator"],
      global_config=self.global_config,
      **orchestrator_args,
    )
    
    return orchestrator

  def _create_workers(self) -> List[mp.Process]:
    print("Creating workers...")
    buffer_size = get_store_args_from_config(self.config["store"])["buffer_size"]

    env_workers = self._create_env_workers()
    actor_workers = self._create_actor_workers()
    store_worker = self._create_store_worker(buffer_size)
    learner_worker = self._create_learner_worker(buffer_size)

    return [
      *env_workers,
      *actor_workers,
      store_worker,
      learner_worker,
    ]

  def _create_env_workers(self) -> List[EnvironmentWorker]:
    print("Creating env workers...")
    
    env_workers = []
    env_worker_args = get_env_args_from_config(self.config["environment"])
    num_workers = env_worker_args["num_workers"]
    num_envs = env_worker_args["num_envs"]
    env_worker_seed_seqs = self.seed_seqs["environment"].spawn(num_workers)
    
    for seed_seq in env_worker_seed_seqs:
      idx, input_queue, output_queue = self.registry.register(
        "environment", 
        env_worker_args, 
        input_maxsize=num_envs,
        output_maxsize=num_envs,
      )
      worker = EnvironmentWorker(
        input_queue=input_queue,
        output_queue=output_queue,
        seed_seq=seed_seq,
        global_config=self.global_config,
        index=idx,
        **env_worker_args,
      )
      env_workers.append(worker)

    return env_workers

  def _create_actor_workers(self) -> List[ActorWorker]:
    print("Creating actor workers...")

    actor_workers = []
    actor_worker_args = get_actor_args_from_config(self.config["actor"])
    num_workers = actor_worker_args["num_workers"]
    actor_worker_seed_seqs = self.seed_seqs["actor"].spawn(num_workers)

    for seed_seq in actor_worker_seed_seqs:
      idx, input_queue, output_queue = self.registry.register("actor", actor_worker_args)
      worker = ActorWorker(
        input_queue=input_queue,
        output_queue=output_queue,
        seed_seq=seed_seq,
        global_config=self.global_config,
        index=idx,
        **actor_worker_args,
      )
      actor_workers.append(worker)

    return actor_workers

  def _create_store_worker(self, buffer_size: int = 0) -> ExperienceStoreWorker:
    print("Creating experience store worker...")

    store_worker_args = get_store_args_from_config(self.config["store"])

    idx, input_queue, output_queue = self.registry.register(
      "store", 
      store_worker_args, 
      output_maxsize=buffer_size
    )
    store_worker = ExperienceStoreWorker(
      input_queue=input_queue,
      output_queue=output_queue,
      seed_seq=self.seed_seqs["store"],
      global_config=self.global_config,
      index=idx,
      **store_worker_args,
    )

    return store_worker

  def _create_learner_worker(self, buffer_size: int = 0) -> LearnerWorker:
    print("Creating learner worker...")

    learner_worker_args = get_learner_args_from_config(self.config["learner"])

    idx, input_queue, output_queue = self.registry.register(
      "learner",
      learner_worker_args,
      input_maxsize=buffer_size,
    )
    learner_worker = LearnerWorker(
      input_queue=input_queue,
      output_queue=output_queue,
      seed_seq=self.seed_seqs["learner"],
      global_config=self.global_config,
      index=idx,
      **learner_worker_args,
    )

    return learner_worker

  def _close_workers(self):
    self.orchestrator.join()
    for worker in self.workers: worker.join()

    self.orchestrator.close()
    for worker in self.workers: worker.close()
