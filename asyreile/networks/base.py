import pathlib
import torch
import torch.nn as nn

from asyreile.core.utils import DEFAULT_CHECKPOINT_DIR, DEFAULT_DEVICE

class BaseNetwork(nn.Module):
  def __init__(
    self, 
    name: str = "network", 
    checkpoint_dir: pathlib.Path = DEFAULT_CHECKPOINT_DIR, 
    device: torch.device = DEFAULT_DEVICE,
    **kwargs
  ):
    super().__init__()
    self.name = name
    self.checkpoint_file = checkpoint_dir/self.name

    self.setup(**kwargs)

    self.device = device
    self.to(self.device)

  def setup(self, **kwargs):
    raise NotImplementedError

  def get_output_shape(self, layer, input_shape):
    return layer(torch.zeros(1, *input_shape)).shape[1:]
  
  def save_checkpoint(self):
    print(f"--- saving checkpoint {self.name} ---")
    torch.save(self.state_dict(), self.checkpoint_file)

  def load_checkpoint(self):
    print(f"--- loading checkpoint {self.name} ---")
    self.load_state_dict(torch.load(self.checkpoint_file))