import gym

class UnitRewardGymWrapper(gym.RewardWrapper):
  def __init__(
    self, 
    env: gym.Env
  ):
    super().__init__(env)
  
  def reward(
    self, 
    reward: float
  ) -> float:
    if reward > 0:
      return 1
    elif reward < 0:
      return -1
    return 0
