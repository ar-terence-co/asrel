from collections import OrderedDict
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional

from asyreile.actors.base import BaseActor
from asyreile.core.utils import get_net_from_config

class GreedyDiscreteActor(BaseActor):
  def __init__(
    self,
    net: Dict = {},
    seed: Optional[int] = None, 
    device: str = "cpu"
  ):
    self.net = get_net_from_config(net)
    self.net.eval()

    self.device = torch.device(device)
    self.generator =  torch.Generator(device=self.device)
    if seed is not None: self.generator.manual_seed(seed)

    self.epsilon = 0

  def choose_action(self, obs: torch.Tensor, greedy: bool = False) -> torch.Tensor:
    with torch.no_grad():
      out = self.net(obs)
      greedy_actions = torch.argmax(out, dim=-1)
      if greedy: return greedy_actions
      
      one_hot = F.one_hot(greedy_actions, num_classes=out.shape[-1])
      eps = torch.full(out.shape, self.epsilon / out.shape[-1])
      probs = one_hot * (1 - self.epsilon) + eps

      actions = torch.multinomial(probs, 1, replacement=True, generator=self.generator).squeeze()
      return actions

  def sync_networks(self, network_params: OrderedDict):
    self.net.load_state_dict(network_params)

  def update(self, **kwargs):
    if "epsilon" in kwargs:
      self.epsilon = kwargs["epsilon"]
    