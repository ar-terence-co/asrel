import gym
import numpy as np
import skimage.color
import skimage.transform
from typing import List, Tuple

class AtariObservationGymWrapper(gym.ObservationWrapper):
  """
  Crop the observation horizontally into a square and scale it to size. 
  """
  def __init__(
    self, 
    env: gym.Env, 
    crop: List[int] = [0, 210],
    size: int = 84,
  ):
    super().__init__(env)
    self.crop = crop
    self.shape = (size, size)

    self.observation_space = gym.spaces.Box(
      low=0., 
      high=1., 
      shape=self.shape, 
      dtype=np.float32
    )

  def observation(
    self, 
    obs: np.ndarray,
  ) -> np.ndarray:
    obs = skimage.color.rgb2gray(obs)
    obs = obs[self.crop[0]:self.crop[1]+1]
    obs = skimage.transform.resize(obs, self.shape)
    return obs
