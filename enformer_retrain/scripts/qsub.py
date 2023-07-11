import subprocess
import os
from os.path import *
import random
from SYSTEM_CONFIG import config_dict
from math import *
import sys

'''
Contains functions to submit training job via qsub.

Can be run directly to submit a single job using kwargs listed below
(loadnev.sh defines the alias qsj for this purpose). This script
can take command line arguments.

qsj -p [run_id] will load a model from a checkpoint. (run_id need not be in
that position, it just needs to be the first valid identifier passed).
Since all system config is stored in the trainer config and all constructor parameters
are stored in the checkpoint, using this option does not require any other arguments
to be passed, and all other arguments will be ignored and replaced with those that were
used by the run for the first job.
'''

kwargs={
    'warmup': '3e3',
    'epochs': '10',
    'learning_rate': "1e-4",
    'lr_schedule': "lwcd",
    'optimizer': "adamw",
    'architecture': 'linear',
    'dataset':'randm',
    'weight_decay': 2e-2,
    'loss': 'mse_loss',
    'metrics': 'poisson_loss',
}

def mkdir(path):
    if not exists(path):
        os.makedirs(path)

def ftot(x):
    '''
    converts float representing length in hours
    to timestamp (hh:mm:ss)
    '''
    mm,hh = modf(x)
    ss,mm = modf(60*mm)
    hh = int(hh)
    mm = int(mm)
    ss = ceil(ss)
    return f'{hh}:{mm}:{ss}'

def generate_id(n):
    '''
    generates random valid identifer string of length n
    
    since wandb is slow to import and the only function used from it
    by run_job is generate_id, I'm just going to define it here
    '''
    chrs = []
    for a,b in [('a','z'),('A','Z'),('0','9')]:
        chrs += [chr(c) for c in range(ord(a),ord(b)+1)]
    first = random.choice(chrs[:52])
    return first + ''.join([random.choice(chrs) for _ in range(n-1)])


def run_job(
    args: list[str],
    resub = False,
    ):
    '''
    Submits a training loop qsub job. Is used by qsub.py's __main__ sequence,
    supersub.py, and train.py (for resubmission).
    
    args:
        args - array of command-line arguments to pass to train.py
        resub - whether or not this job is a resubmission of another that has hit
            a soft timeout. Should only be enabled when calling from train.py

    args is kept as an array because we can simply have `run_job(sys.argv[1:])` at the
    end of train.py. The array is also more convenient for the preprocessing we will do
    in this function.
    
    If '-D' is included, it will be expanded to every debug flag (see train.py for information on flags)

    Including '-p' causes train.py to ignore all non-flag arguments.

    run_job will use qalter to pipe output into enformer_retrain/logs/{run_id}.out, set the hard runtime
    (the point at which qsub itself will timeout) to 30 minutes greater than the script runtime, 
    and change the job's name to match run_id.
    
    '''
    this_dir = os.path.dirname(__file__)
    enformer_retrain = join(this_dir,'..')
    for d in ['logs','checkpoint','save']:
        mkdir(join(enformer_retrain,d))

    if '-D' in args:
        args.remove('-D')
        for f in ['n','q','t','w','c']:
            args.append('-'+f)
    if '-n' in args:
        cmd='bash'
    else:
        cmd='qsub'

    config_str = ""
    for cdict in config_dict.values():
        for kw,arg in cdict.items():
            config_str+=f"--{kw}={arg} "
    if not exists(join(this_dir,'wandb_project.txt')):
        print('Please add wandb_project.txt to enformer_retrain/scripts/')
        exit()
    with open(join(this_dir,'wandb_project.txt')) as fp:
        config_str+=f"--WANDB_PROJECT={fp.readline()}"

    if not resub:
        userargs = args
        args = [None for _ in range(3)]
        args[0] = enformer_retrain
        args[1] = 0
        if '-p' not in userargs:
            args[2] = generate_id(7)
        else:
            for i, arg in range(userargs):
                if arg.isidentifier():
                    args[2] = userargs.pop(i)
        args.extend(userargs)
    
    args[1] = str(args[1])

    # will be used later
    qsub_count = args[1]
    run_id = args[2]
    
    argstr = f"{' '.join(args)} {config_str}"


    executable=join(this_dir,'job.sh')
    if '-n' in args:
        cmd = f"{cmd} {executable} {argstr}"
    else:
        log_path = join(this_dir,'..','logs',f'{run_id}.out')
        SYSTEM_TIMEOUT = ftot(ceil(config_dict['trainer']['TIMEOUT'])+0.5) # timeout plus half an hour
        resources = config_dict['qsub_resources'].items()
        resources = ','.join([f'{r}={q}' for r,q in resources])
        resources = f'{resources},h_rt={SYSTEM_TIMEOUT}'
        qsub_options = f'-N {run_id}_{qsub_count} -o {log_path} -e {log_path} -l {resources}'
        cmd = f"{cmd} {qsub_options} {executable} {argstr}"
    
    print(f"======QSUB.PY======")
    print(f"RUN ID {run_id}")
    print(cmd)
    os.system(cmd)
if __name__=='__main__':
    '''
    qsub.py may be run on it's own to submit a single job. Uses the same arguments and
    same command line format as supersub.py
    '''
    args = [f"--{keyword}={arg}" for keyword,arg in kwargs.items()]
    args.extend(sys.argv[1:])
    run_job(args)

