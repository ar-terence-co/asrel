import copy
from gym import Space
import numpy as np
import time
import torch
import torch.nn.functional as F
from typing import Dict

from asyreile.core.utils import get_net_from_config
from asyreile.learners.base import BaseLearner

class DQNLearner(BaseLearner):
  def __init__(
    self, 
    net: Dict = {},
    epsilon_start = 1.,
    epsilon_dec = 1e-4,
    epsilon_end = 1e-3,
    **kwargs
  ):
    super().__init__(**kwargs)

    self.net = get_net_from_config(
      net,
      input_size=self.input_space.shape,
      output_size=(self.output_space.n,),
      device=self.device,
      checkpoint_dir=self.checkpoint_dir
    )
    self.target_net = copy.deepcopy(self.net)
    self.target_net.requires_grad_(False)

    self.n_steps = self.global_config.get("n_steps", 1)
    self.gamma = self.global_config.get("gamma", 0.99)
    self.epsilon = self.global_config.get("epsilon", 1.)
    self.epsilon_dec = self.global_config.get("epsilon_dec", 1e-4)
    self.epsilon_end = self.global_config.get("epsilon_end", 1e-3)

    self.total_steps = 0

  def train(self):
    for data in self.data_stream:
      self.net.optimizer.zero_grad()

      state = data["state"]
      action = data["action"]
      ret = data["return"]
      nth_state = data["nth_state"]
      done = data["done"]
      
      batch_index = np.arange(state.shape[0], dtype=np.int32)

      q_val = self.net(state)
      q_val = self.net(state)[batch_index, action.to(torch.long)]

      q_next = self.target_net(nth_state)
      q_next[done] = 0.0
      q_target = ret.float() + (self.gamma**self.n_steps) * torch.max(q_next, dim=-1)[0]

      loss = F.mse_loss(q_val, q_target)
      loss.backward()
      self.net.optimizer.step()
    
      self.total_steps += 1
      