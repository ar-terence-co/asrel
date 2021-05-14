import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List

from asrel.core.utils import get_instance_from_config
from asrel.networks.base import BaseNetwork

class SimpleConv2DNetwork(BaseNetwork):
  def setup(
    self,
    input_size: List[int],
    output_size: List[int],
    conv_params: List[Dict] = [
      {"out_channels": 32, "kernel_size": 8, "stride": 4},
      {"out_channels": 64, "kernel_size": 4, "stride": 2},
      {"out_channels": 64, "kernel_size": 3, "stride": 1},
    ],
    ff_layers: List[int] = [128],
    activation: Dict = {},
    optimizer: Dict = {},
  ):
    conv_layers = []
    for i, params in enumerate(conv_params):
      in_channels = input_size[0] if i == 0 else conv_params[i-1]["out_channels"]
      conv = nn.Conv2d(in_channels, **params)
      activ = get_instance_from_config(nn, **activation, default_class=nn.ReLU)
      seq = nn.Sequential(conv, activ)
      conv_layers.append(seq)
    
    self.conv_block = nn.Sequential(
      *conv_layers,
      nn.Flatten(),
    )
    conv_output_shape = self.get_output_shape(self.conv_block, input_size)[0]
    layers = []
    for i, layer_size in enumerate(ff_layers):
      prev_size = conv_output_shape if i == 0 else ff_layers[i-1]
      linear = nn.Linear(prev_size, layer_size)
      activ = get_instance_from_config(nn, **activation, default_class=nn.ReLU)
      seq = nn.Sequential(linear, activ)
      layers.append(seq)
    
    self.ff_block = nn.Sequential(*layers)
    self.out_layer = nn.Linear(ff_layers[-1], output_size[0])

    self.optimizer = get_instance_from_config(
      optim, 
      params=self.parameters(), 
      default_class=optim.Adam, 
      **optimizer
    )

  def forward(self, x: torch.Tensor):
    x = x.float()
    x = self.conv_block(x)
    x = self.ff_block(x)
    out = self.out_layer(x)
    return out