import numpy as np
from asrel.core.runner import Runner
from asrel.core.utils import get_args

if __name__ == "__main__":
  args = get_args()

  # if args.test_env_worker:
  #   from asrel.core.workers.tests import test_environment_worker
  #   test_environment_worker(config, seeds)
  #   sys.exit()

  # if args.test_actor_worker:
  #   from asrel.core.workers.tests import test_actor_worker
  #   test_actor_worker(config, seeds)
  #   sys.exit()

  asrel_runner = Runner(args.config)
  asrel_runner.start()
