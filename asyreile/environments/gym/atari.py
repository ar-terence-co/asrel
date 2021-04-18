import gym

from asyreile.environments.base import BaseEnvironment

class AtariEnvironment(gym.Wrapper, BaseEnvironment):
  def __init__(self, id: str, **kwargs):
    env = gym.make(id)
    super().__init__(env)

    self.env = env