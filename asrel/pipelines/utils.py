from multiprocessing.queues import Queue, Full
from typing import Any, Callable

def put_while(
  queue: Queue, 
  task: Any, 
  predicate: Callable[[], bool], 
  timeout: int = 3
):
  while predicate():
    try:
      queue.put(task, block=True, timeout=timeout)
      break
    except Full:
      pass
