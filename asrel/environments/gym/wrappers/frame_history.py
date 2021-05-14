import gym
import numpy as np
from typing import Dict, Tuple

class FrameHistoryGymWrapper(gym.Wrapper):
  """
  Return the last n frames as a single observation.
  This is used to get information on velocity.
  """
  def __init__(
    self, 
    env: gym.Env, 
    history_len: int = 4
  ):
    super().__init__(env)
    self.env = env
    self.history_len = history_len
    self.observation_space.shape = (self.history_len, *self.observation_space.shape)
  
  def reset(self) -> np.ndarray:
    obs = self.env.reset()
    self.history = np.repeat(
      np.expand_dims(obs, axis=0), 
      self.history_len, 
      axis=0,
    )
    return self.history

  def step(
    self, 
    action: np.ndarray
  ) -> Tuple[np.ndarray, float, bool, Dict]:
    obs, reward, done, info = self.env.step(action)
    self.history = np.append(
      self.history[:-1],
      np.expand_dims(obs, axis=0),
      axis=0,
    )
    return self.history, reward, done, info
