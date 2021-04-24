from queue import Queue
import torch
from threading import Thread

from asyreile.core.utils import get_args, get_config, get_seed_sequences

def setup(config):
  config = get_config()

  if "seed" in config:
    config["seed"]
  torch_seed = get_seed("torch")

if __name__ == "__main__":
  args = get_args()
  config = get_config(args.config)
  seeds = get_seed_sequences(config)

  # setup()

  if args.test_env_worker:
    from asyreile.core.workers.tests import test_environment_worker
    test_environment_worker(config, seeds)
