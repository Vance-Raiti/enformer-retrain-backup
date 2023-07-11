import torch.optim as optim
import torch.optim.lr_scheduler as schedule
import math

#
# File containing lr schedule functions for LambdaLR schedules
#
# A set of functions that return another 
# function parameterized by the total training length
# and the warmup duration. The function will then be
# passed to torch.optim.lr_scheduler.LambdaLR as the
# lambda function
#

def no_schedule(*args,**kwargs):
    return lambda n: 1

def linear_warmup_linear_decay(warmup, N, **kwargs):
    def lwld(n):
        if n < warmup:
            return n/warmup
        return 1-(n-warmup)/(N-warmup)
    return lwld


def linear_warmup_cosine_decay(warmup, N, **kwargs):
    def lwcd(n):
        if n < warmup:
            return n/warmup
        theta = math.pi*(warmup-n)/(N-warmup)
        return 0.5*math.cos(theta)+0.5
    return lwcd

def linear_warmup_stepdown(warmup, N, cooldown,**kwargs):
    def lws(n):
        if n > (N-cooldown):
            return 0.2
        return min(n/warmup,1)
    return lws


Schedules = {
    'name': 'Schedules',
    "no_schedule": no_schedule,
    "lwcd": linear_warmup_cosine_decay,
    "lwld": linear_warmup_linear_decay,
    "lws": linear_warmup_stepdown,
}


