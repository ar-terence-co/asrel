import torch
import torch.nn as nn

def soft_update(
  source: nn.Module, 
  target: nn.Module, 
  tau: float = 1e-4
):
  for source_param, target_param in zip(source.parameters(), target.parameters()):
    target_param.data.copy_(tau * source_param.data + (1. - tau) * target_param.data)

def hard_update(
  source: nn.Module,
  target: nn.Module
):
  for source_param, target_param in zip(source.parameters(), target.parameters()):
    target_param.data.copy_(source_param.data)