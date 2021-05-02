import numpy as np
from typing import Dict

from asyreile.replays.base import BaseReplay


class StandardReplay(BaseReplay):
  def __init__(
    self,
    batch_size: int = 256,
    maxsize: int = 100000,
    types: Dict[str, str] = {},
    **kwargs,
  ):
    self.batch_size = batch_size
    self.maxsize = maxsize
    self.types = types

    self.cursor = 0
    self.size = 0
    self.total_count = 0

    self.replay_dict = None    

  def store(self, experience: Dict):
    if not self.replay_dict:
      self.replay_dict = self._create_replay_dict(experience)

    for name, value in experience.items():
      self.replay_dict[name][self.cursor] = value

    self.cursor = (self.cursor+1) % self.maxsize
    if self.size < self.maxsize: self.size += 1
    self.total_count += 1

  def sample(self) -> Dict[str, np.ndarray]:
    if not self.replay_dict: return {}
    batch = np.random.choice(self.size, self.batch_size)
    batch_experience = self[batch]

    return batch_experience

  def _create_replay_dict(self, experience: Dict):
    replay_dict = {}
    for key, value in experience.items():
      is_np_instance = isinstance(value, np.ndarray)
      shape = value.shape if is_np_instance else ()

      if key in self.types:
        type_ = np.dtype(self.types[key])
      elif is_np_instance:
        type_ = value.dtype
      else:
        type_ = np.array(value).dtype

      replay_dict[key] = np.zeros((self.maxsize, *shape), dtype=type_)

    return replay_dict

  def __getitem__(self, idx):
    return {
      key: exps[idx]
      for key, exps in self.replay_dict.items()
    }
