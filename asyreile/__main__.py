from queue import Queue
from threading import Thread

from asyreile.core.utils import get_args, get_config


if __name__ == "__main__":
  args = get_args()
  config = get_config(args.config)

  if args.test_env_worker:
    from asyreile.core.workers.tests import test_environment_worker
    test_environment_worker(config)
