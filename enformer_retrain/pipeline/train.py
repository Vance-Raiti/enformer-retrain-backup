'''
loads objects from parameters, prepares environment, and calls into 
trainer.Trainer.train(). If trainer hits a soft timeout, resubmit
the checkpointed job

args: (keywords indicated with a (s) indicate that they will be used to select from a dictionary
      defined within one of the/*.py)
    learning_rate - base learning rate
    lr_schedule (s) - name of lr schedule


'''


import torch
import wandb

# User-defined objects for training
from trainer import Trainer, TrainerConfig
from enformer_retrain.modules import *

# other utilities
import argparse
from enformer_retrain.utils import *
from enformer_retrain.scripts import run_job, config_dict
import sys
import os
import os.path as path
from os.path import join



parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate",default=1.0,type=float)
parser.add_argument("--lr_schedule",default='no_schedule',type=str)
parser.add_argument("--epochs",default=1,type=int)
parser.add_argument("--warmup",type=float) # for lr schedules
parser.add_argument("--optimizer",default="adam",type=str)
parser.add_argument("--cooldown",type=float) # for lr schedules
parser.add_argument("--loss",default='poisson_loss',type=str)
parser.add_argument("--metrics",type=str) # can be comma-separated list
parser.add_argument("--architecture",default='enformer',type=str) # model architecture (e.g. Enformer, chop)
parser.add_argument('--weight_decay',type=float) # for weight decay optimizers
parser.add_argument('-p','--checkpointed',default=False,action='store_true')
parser.add_argument('--dataset',default='basenji')



# for debugging on interactive shell
# flags are listed in SYSTEM_CONFIG.
# It feels a little 
# the arg parser as well as passed on to future
# calls to resub.sh

flags = {
    '-n':'--no_qsub', # ignored, used by qsub scripts
    '-t':'--short_train',
    '-w':'--no_wandb',
    '-q':'--quick_timeout',
    '-c':'--frequent_checkpoint',
}
for f,flag in flags.items():
    parser.add_argument(f,flag,default=False,action='store_true')

for cdict in config_dict.values():
    for kw in cdict:
        parser.add_argument(f'--{kw}')
# training requires two directories, checkpoint (for intermediate checkpoints),
# and save (for the final model)
# Requires the user to specify their wandb project explicitly
# wandb_project.txt is listed in .gitignore
# --- PARSING COMMAND LINE ---
parser.add_argument('--WANDB_PROJECT')
# $1: path of enformer_retrain/scripts/ (ignored)
# $2: qsub count (ignored)
# $3: id
# $[4:]: kwargs
print('========train.py==============')
print('sys.argv:')
print(sys.argv)
run_id = sys.argv[3]

if '-w' in sys.argv:
    os.environ['WANDB_MODE'] = 'disabled'
else:
    os.environ['WANDB_MODE'] = 'online'
    os.environ['WANDB_SILENT']='true'
    os.environ['WANDB_CONSOLE']='off'

args = parser.parse_args(sys.argv[4:])

if '-q' in sys.argv:
    TIMEOUT = args.QUICK_TIMEOUT
else:
    TIMEOUT = args.TIMEOUT

if '-t' in sys.argv or '-q' or '-n' in sys.argv:
    PRINT_INTERVAL = 1
else:
    PRINT_INTERVAL = args.PRINT_INTERVAL

if '-f' in sys.argv:
    CHECKPOINT_INTERVAL = args.FREQUENT_CHECKPOINT_INTERVAL
else:
    CHECKPOINT_INTERVAL = args.CHECKPOINT_INTERVAL

if '-t' in sys.argv:
    SHORT_TRAIN_ITERATIONS = args.SHORT_TRAIN_ITERATIONS
else:
    SHORT_TRAIN_ITERATIONS = None

WANDB_PROJECT = args.WANDB_PROJECT

print(f"timeout: {TIMEOUT}, print_interval: {PRINT_INTERVAL}, checkpoint_inteval: {CHECKPOINT_INTERVAL}")

TIMEOUT = float(TIMEOUT)
PRINT_INTERVAL = int(PRINT_INTERVAL)
CHECKPOINT_INTERVAL = float(CHECKPOINT_INTERVAL)
if SHORT_TRAIN_ITERATIONS is not None:
    SHORT_TRAIN_ITERATIONS = int(SHORT_TRAIN_ITERATIONS)
# Enformer is huge. This is fixed for now
batch_size = 1

# Since warmup period is probably going to be long, I want to let
# It be expressed using floating-point notation even though it's an int

if args.warmup is not None:
    args.warmup = int(args.warmup) 
if args.cooldown is not None:
    args.cooldown = int(args.cooldown)
warmup = args.warmup

lrs = args.lr_schedule
optimizer = args.optimizer
learning_rate = args.learning_rate




# --- INITIALIZING DATASETS ---
'''
I'm not sure if basenji supports random access, so I'm just going to have us
initialize a new dataset every job. Inside of Trainer.run_epoch it will skip over
all of the datapoints we've already done for that split of that epoch.
'''
if 'basenji' in args.dataset:
    kwargs = {
        'organism': 'human',
        'seq_length': 196608
    }
else:
    kwargs = {
        'size': 4,
    }
dataset = Datasets[args.dataset]
train_data, validation_data = [dataset(**{'split':split,**kwargs}) for split in ['train','valid']]
print(len(train_data))
# --- TRAINING OBJECTS ---
if not args.checkpointed:
    model = select(
        name = args.architecture,
        kwargs = {
            'dim': 1536,
            'depth': 11,
            'heads': 8,
            'output_heads': dict(human = 5313),
            'use_checkpointing': True,
            'out_shape': (1,896,5313),
        },
        dictionary = Models,
    ).to(device())
    
    optimizer = select(
        name = args.optimizer, 
        kwargs = {
            'lr': learning_rate,
            'weight_decay': args.weight_decay,
        },
        objs = {
            'params': model.parameters(),
        },
        dictionary = Optimizers,
    )
    
    training_schedule = select(
        name = args.lr_schedule,
        kwargs = {
            'warmup': args.warmup,
            'N': args.epochs*len(train_data),
            'cooldown': args.cooldown,
            'lr_schedule': args.lr_schedule,
        },
        objs = {
            'optimizer': optimizer,
        },
        dictionary = TrainingSchedules,
    )
    

    metrics = {}
    metrics[args.loss] = None
    for m in args.metrics.split(','):
        metrics[m] = None
    for m in metrics:
        metrics[m] = select(
            name = m,
            kwargs = {
                'n_channels': 5313,
                'is_differentiable': (m==args.loss),
            },
            dictionary = Metrics,
        )

    # we don't need a fancy function to prepare the config. It's just
    # a container

    # lists which/what order of params you would like to appear in the 
    # wandb title/save name
    key_params = ['architecture','lr_schedule','learning_rate','optimizer']
    wandb_name = ""
    for kp in key_params:
        wandb_name += f"{getattr(args,kp)}, "
    wandb_name = wandb_name[:-2]
    
    # along with the current epoch
    save_name = run_id
    wandb_config = {}
    for kw, arg in vars(args).items():
        wandb_config[kw]=arg
    wandb_config['TIMEOUT'] = TIMEOUT
    save_name = save_name[:-1]
    wandb_config['id'] = run_id
    wandb_config['working directory']=os.path.dirname(__file__)
    wandb_kwargs = {
        'name': wandb_name,
        'config': wandb_config,
        'project': WANDB_PROJECT,
    }

    # collect config flags to be passed to trainer
    config_flags = {}
    for flag in flags.values():
        flag = flag[2:] # strip of leading '--'
        config_flags[flag] = getattr(args,flag)
    config = TrainerConfig(
        max_grad_norm = 0.2, #
        dloader_workers = 0,
        batch_size = 1,
        save_name = save_name,
        wandb_kwargs = wandb_kwargs,
        flags = config_flags,
        loss = args.loss,
        run_id = run_id,
        TIMEOUT = TIMEOUT,
        PRINT_INTERVAL = PRINT_INTERVAL,
        CHECKPOINT_INTERVAL = CHECKPOINT_INTERVAL,
        SHORT_TRAIN_ITERATIONS = SHORT_TRAIN_ITERATIONS,
    )

else: 
    # save dictionary
    script_dir = os.path.dirname(__file__)
    load_path= join(script_dir,'checkpoint',run_id+'.pt')
    sd = torch.load(load_path,map_location='cpu')
    
    # under the hood torch uses pickle for save and load
    config = sd['config'] # I'm pretty sure torch is fine saving regular objects?
    
    # convention for labeling a particular object's blueprint must be consistent between here and
    # trainer.Trainer.train()
    #
    # For now, the convention will be 'label-bp'[0] = object type and 'label-bp'[1] = kwargs for object initializer
    # note that the actual instance of the model, optimizer, and lr schedule cannot be saved via torch.save, so
    # we must explicitly initialize them and append them to kwargs
    
    # set the blueprint attribute of an object. Copies kwargs directly from save dictionary

    model = ld(
        label='model',
        sd=sd,
        dictionary=Models,
    ).to(device())
    
    optimizer = ld(
        label='optimizer',
        sd=sd,
        dictionary=Optimizers,
        objs={
            'params': model.parameters(),
        }
    )
    
    training_schedule = ld(
        label='training_schedule',
        sd=sd,
        dictionary=TrainingSchedules,
        objs={
            'optimizer': optimizer,
        },
    )
    metrics = {}
    for m in sd['metrics']:
        metrics[m] = ld(
            label=m,
            sd=sd,
            dictionary=Metrics
        )

train_obj = Trainer(
    model = model,
    config = config,
    optimizer = optimizer,  
    training_schedule = training_schedule,
    train_data = train_data,
    validation_data = validation_data,
    metrics = metrics,
)
# Trainer.train() will return
# Trainer.status() is COMPLETE if epochs == epoch
# otherwise it is INCOMPLETE
train_obj.train()
if training_schedule.done:
    print('train.py: Exiting')
    exit()
print('train.py: running run_job')

args = sys.argv[1:]
if '-p' not in args:
    args.append('-p')

run_job(args,resub=True)



