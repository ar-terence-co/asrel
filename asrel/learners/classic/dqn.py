import copy
from gym import Space
import numpy as np
import time
import torch
import torch.nn.functional as F
from typing import Dict

from asrel.core.utils import get_net_from_config
from asrel.learners.base import BaseLearner
from asrel.learners.utils import hard_update, soft_update

class DQNLearner(BaseLearner):
  def __init__(
    self, 
    net: Dict = {},
    epsilon: float = 1.,
    epsilon_dec: float = 1e-4,
    epsilon_end: float = 1e-3,
    use_hard_update: bool = False,
    hard_update_freq: int = 1000,
    soft_update_tau: float = 5e-3,
    policy_update_freq: int = 1000,
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

    self.epsilon = epsilon
    self.epsilon_dec = epsilon_dec
    self.epsilon_end = epsilon_end

    self.use_hard_update = use_hard_update
    self.hard_update_freq = hard_update_freq
    self.soft_update_tau = soft_update_tau

    self.policy_update_freq = policy_update_freq

    self.total_steps = 0

    self._update_policy()

  def train(self):
    for data in self.data_stream:
      self.net.optimizer.zero_grad()
      loss = self._compute_loss(data)
      loss.backward()
      self.net.optimizer.step()
    
      self._update_target()
      self._update_policy()
      
      self.total_steps += 1

  def _compute_loss(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
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
    return loss

  def _update_target(self):
    if self.use_hard_update:
      if self.total_steps % self.hard_update_freq == 0:
        hard_update(self.net, self.target_net)
    else:
      soft_update(self.net, self.target_net, self.soft_update_tau)

  def _update_policy(self):
    if self.epsilon > self.epsilon_end:
      self.epsilon = max(self.epsilon - self.epsilon_dec, self.epsilon_end)
      self.send_actor_update({"epsilon": self.epsilon})

    if self.total_steps % self.policy_update_freq == 0:
      self.send_network_update({
        "net": self.net.state_dict()
      })

      