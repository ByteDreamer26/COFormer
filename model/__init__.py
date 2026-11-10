import argparse
from torch import nn
from typing import Union, Dict, Any
from model.llama import TrajLlama, CriticLlama
from model.transformer_xl import TransformerXL

class UniCO(nn.Module):
    def __init__(self, config: Union[argparse.Namespace, Dict[str, Any]]):
        super(UniCO, self).__init__()
        self.config = config

        if config.model == 'llama':
            self.transformer = TrajLlama(config)
        elif config.model == 'transformer_xl':
            self.transformer = TransformerXL(config)
        else:
            raise NotImplementedError

    def forward(self, *args, **model_inputs):
        model_inputs = model_inputs or {}
        return self.transformer(*args, **model_inputs)
    
class UniCO_CRITIC(nn.Module):
    def __init__(self, config: Union[argparse.Namespace, Dict[str, Any]]):
        super(UniCO_CRITIC, self).__init__()
        self.config = config

        if config.model == 'llama':
            self.transformer = CriticLlama(config)
        else:
            raise NotImplementedError

    def forward(self, *args, **model_inputs):
        model_inputs = model_inputs or {}
        return self.transformer(*args, **model_inputs)