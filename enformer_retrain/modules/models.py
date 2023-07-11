from .enformer_pytorch.modeling_enformer import Enformer
import torch


# this is just here for pipeline testing
class Chop(torch.nn.Module):
    def __init__(self,out_shape,**kwargs):
        super().__init__()
        self.n = 1
        self.out_shape = out_shape
        self.linear = torch.nn.Linear(in_features=4,out_features=1)
        for dim in out_shape:
            self.n*=dim
        
    def forward(self,x):
        x = self.linear(x)
        x = torch.flatten(x)
        if x.shape[0] < self.n:
            z = torch.flatten(torch.zeros(device = x.device, dtype=x.dtype, size=(self.n-x.shape[0],1))) 
            x = torch.cat((x,z))
        if x.shape[0] > self.n:
            x = x[:self.n]
        return torch.reshape(x,self.out_shape)

Models = {
    'name': 'Models',
    'enformer': Enformer.from_hparams,
    'linear': lambda *args, **kwargs: torch.nn.Linear(in_features=4,out_features=4),
    'chop': Chop,
}
