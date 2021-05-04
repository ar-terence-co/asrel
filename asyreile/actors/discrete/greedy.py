from collections import OrderedDict
from gym import Space
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional

from asyreile.actors.base import BaseActor
from asyreile.core.utils import get_net_from_config

class GreedyDiscreteActor(BaseActor):
  def __init__(
    self,
    net: Dict = {},
    **kwargs
  ):
    super().__init__(**kwargs)

    self.net = get_net_from_config(
      net,
      input_size=self.input_space.shape,
      output_size=(self.output_space.n,),
      device=self.device,
      checkpoint_dir=self.checkpoint_dir,
    )
    self.net.requires_grad_(False)
    self.net.eval()

    self.epsilon = 0

  def choose_action(self, obs: torch.Tensor, greedy: bool = False) -> torch.Tensor:
    out = self.net(obs)
    greedy_actions = torch.argmax(out, dim=-1)
    if greedy: return greedy_actions
    
    one_hot = F.one_hot(greedy_actions, num_classes=out.shape[-1])
    eps = torch.full(out.shape, self.epsilon / out.shape[-1]).to(self.device)
    probs = one_hot * (1 - self.epsilon) + eps

    actions = torch.multinomial(probs, 1, replacement=True).view(-1)
    return actions

  def sync_networks(self, state_dicts: Dict[str, OrderedDict]):
    if "net" in state_dicts:
      self.net.load_state_dict(state_dicts["net"])

  def update(self, **kwargs):
    if "epsilon" in kwargs:
      self.epsilon = kwargs["epsilon"]
    