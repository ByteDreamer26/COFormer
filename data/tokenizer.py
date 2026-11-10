import torch
import numpy as np
import torch

class ContinuousScalarTokenizer:
    ''' Continous scalar tokenizer, Adapted from DB1's implementation '''
    def __init__(
        self, ver:str='v1', num_continuous_bin:int=1024, mu:float=100.0, M:float=256.0
    ):
        self.num_continuous_bin = num_continuous_bin
        self.ver = ver
        self.mu = mu
        self.M = M

    def discretize(self, x, is_action: bool):
        """ Discretization of float scalars, if is_action then don't need mu-law scaling. """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.copy()).float()
        assert is_action == False, 'The action of the COP is an integer'
        
        x_mu_lawed = (
            torch.sign(x)
            * torch.log(torch.abs(x) * self.mu + 1.0)
            / torch.log(torch.tensor(self.mu * self.M + 1.0))
        )

        x = x_mu_lawed
        if self.ver == 'v1':    
            # 在 [-1, 1] 范围内进行离散化
            #x = torch.clamp(x_mu_lawed, -1, 1)
            assert -1-1e-2 <= x.min() <= x.max() <= 1+1e-2
            x = ((x + 1) / 2 * self.num_continuous_bin).int()
            x = torch.clamp(x, 0, self.num_continuous_bin - 1).int()
        elif self.ver == 'v2':
            # 在 [0, 1] 范围内进行离散化
            assert 0-1e-2 <= x.min() <= x.max() <= 1+1e-2
            x = (x * self.num_continuous_bin).int()
            x = torch.clamp(x, 0, self.num_continuous_bin - 1).int()
        else:
            raise NotImplementedError
        return x

    def decode(self, x, is_action: bool):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if x.max() >= self.num_continuous_bin or x.min() < 0:
            print(
                "Warning of exceeded range of discrete number to recontruct, "
                "by default values will be cliped, min: {}, max:{}".format(
                    x.min(), x.max()
                )
            )
            x = np.clip(x, 0, self.num_continuous_bin - 1)

        x = (x.float() / self.num_continuous_bin) * 2 - 1
        if not is_action:
            x = torch.sign(x) * ((1 + self.M * self.mu) ** torch.abs(x) - 1) / self.mu

        return x