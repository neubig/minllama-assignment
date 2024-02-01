from dataclasses import dataclass

import re
from torch import dtype
from config import LlamaConfig
from utils import *

class LlamaPreTrainedModel(nn.Module):
  config_class = LlamaConfig
  base_model_prefix = "llama"

  def __init__(self, config: LlamaConfig):
      super().__init__()
      self.config = config
      self.vocab_size = config.vocab_size
      self.n_layers = config.n_layers

  def init_weights(self):
    # Initialize weights
    self.apply(self._init_weights)

  def _init_weights(self, module):
    """ Initialize the weights """
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  @property
  def dtype(self) -> dtype:
    return get_parameter_dtype(self)
