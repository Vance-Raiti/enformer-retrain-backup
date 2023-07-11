from qsub import run_job
import sys

'''
Submits a job for every possible combination of kwargs
listed below

args:
    None (all flags will be ignored)
'''


kwargs={
    'warmup': [9e4],
    'epochs': [20],
    'learning_rate': [1e-5,5e-5,1e-4],
    'lr_schedule': ["lwcd"],
    'optimizer': ["adamw"],
    'weight_decay':[1e-2,2e-2,4e-2],
    'metrics': ['mean_corr_coef'],
    'loss': ['poisson_loss'],
}

idxs = [0 for _ in kwargs]
n = 0
while n < len(idxs):
    args = [] #command line arguments
    for idx, kw in zip(idxs, kwargs):
        args.append(f"--{kw}={kwargs[kw][idx]}")
    run_job(args)
    carry = 1
    n = 0
    while n<len(idxs) and carry == 1:
        idxs[n]+=1
        if idxs[n] == len(list(kwargs.values())[n]):
            idxs[n] = 0
            n += 1
        else:
            carry = 0


