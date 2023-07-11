from abc import ABC, abstractmethod

import torch
from .lr_schedules import Schedules

def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr

class TrainingSchedule(ABC):

    @abstractmethod
    def step(self) -> None:
        pass

    @property
    @abstractmethod
    def done(self) -> bool:
        pass

    @abstractmethod
    def state_dict(self) -> dict:
        pass

    @abstractmethod
    def load_state_dict(self) -> None:
        pass

class LambdaLR(TrainingSchedule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        N: int,
        lr_schedule: str,
        **kwargs,
    ):
        global Schedules
        self.optimizer = optimizer
        self.n = 0
        self.N = N
        assert len(optimizer.param_groups) == 1, 'this class isn\'t really made for optimizer with multiple param groups'
        self.lr = optimizer.param_groups[0]['lr']
        self.last_lr = self.lr

        self.lr_schedule = Schedules[lr_schedule](N=N,**kwargs)
        

    def step(self):
        self.n += 1
        scale = self.lr_schedule(self.n)
        
        self.last_lr = scale*self.lr
        set_lr(self.optimizer,self.last_lr)
    @property
    def done(self):
        return self.n >= self.N
    
    def state_dict(self):
        sdict = {
            'n': self.n,
            'lr': self.lr,
        }
        return sdict

    def load_state_dict(self,sdict):
        self.n = sdict['n']
        self.lr = sdict['lr']


TrainingSchedules = {
}

for lr_schedule in Schedules:
    TrainingSchedules[lr_schedule] = lambda *args, **kwargs: LambdaLR(*args,**kwargs)

