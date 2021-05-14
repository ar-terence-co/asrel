import gym
import numpy as np
from typing import Dict, Tuple

class TotalScoreGymWrapper(gym.Wrapper):
  """
  Accumulate the actual score of the episode
  """
  def __init__(
    self, 
    env: gym.Env,
  ):
    super().__init__(env)
    self.env = env
    self.score = 0

  def reset(self) -> np.ndarray:
    self.score = 0
    return self.env.reset()
  
  def step(
    self, 
    action: np.ndarray
  ) -> Tuple[np.ndarray, float, bool, Dict]:
    next_state, reward, done, info = self.env.step(action)
    self.score += reward
    info["score"] = self.score
    return next_state, reward, done, info
