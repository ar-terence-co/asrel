import torch
import torch.nn as nn
import torch.optim as optim
from typing import List

class FeedForwardNetwork(nn.Module):
  def __init__(
    self,
    input_size: List[int],
    output_size: List[int],
    hidden_layers: List[int] = [256, 256],
  ):
    super().__init__()
    layer_dims = [input_size[0]] + hidden_layers
    layers = []
    for i in range(len(layer_dims)-1):
      linear = nn.Linear(layer_dims[i], layer_dims[i+1])
      activ = nn.ReLU()
      seq = nn.Sequential(linear, activ)
      layers.append(seq)
    self.ff_layers = nn.ModuleList(layers)
    self.out_layer = nn.Linear(layer_dims[-1], output_size[0])

    self.optimizer = None

  def forward(self, x: torch.Tensor):
    x = x.float()
    for layer in self.ff_layers:
      x = layer(x)
    out = self.out_layer(x)
    return out