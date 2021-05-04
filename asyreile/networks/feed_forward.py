import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List

from asyreile.core.utils import get_instance_from_config
from asyreile.networks.base import BaseNetwork

class FeedForwardNetwork(BaseNetwork):
  def setup(
    self,
    input_size: List[int],
    output_size: List[int],
    hidden_layers: List[int] = [256, 256],
    activation: Dict = {},
    optimizer: Dict = {},
  ):
    layer_dims = [input_size[0]] + hidden_layers
    layers = []
    for i in range(len(layer_dims)-1):
      linear = nn.Linear(layer_dims[i], layer_dims[i+1])
      activ = get_instance_from_config(nn, **activation, default_class=nn.ReLU)
      seq = nn.Sequential(linear, activ)
      layers.append(seq)
    self.ff_layers = nn.ModuleList(layers)
    self.out_layer = nn.Linear(layer_dims[-1], output_size[0])

    self.optimizer = get_instance_from_config(
      optim, 
      params=self.parameters(), 
      default_class=optim.Adam, 
      **optimizer
    )

  def forward(self, x: torch.Tensor):
    x = x.float()
    for layer in self.ff_layers:
      x = layer(x)
    out = self.out_layer(x)
    return out