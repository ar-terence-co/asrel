import gym
import numpy as np
from typing import Dict, Tuple

class FrameSkipGymWrapper(gym.Wrapper):
  """
  Each step sends the same action for n frames. 
  Returns the last observation.
  """
  def __init__(
    self, 
    env: gym.Env, 
    skip_len: int = 4
  ):
    super().__init__(env)
    self.env = env
    self.skip_len = max(skip_len, 1)
  
  def step(
    self, 
    action: np.ndarray
  ) -> Tuple[np.ndarray, float, bool, Dict]:
    total_reward = 0
    for _ in range(self.skip_len):
      obs, reward, done, info = self.env.step(action)
      total_reward += reward
      if done: break
    return obs, total_reward, done, info