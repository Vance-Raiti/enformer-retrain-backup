import sys
import time
import torch
import wandb
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.cuda.amp import autocast, GradScaler
import os
from os.path import *
import torch.optim as optim
import time
from enformer_retrain.modules.training_schedules import TrainingSchedule
from enformer_retrain import PATH as root

from typing import Optional
from torchmetrics import Metric

from enformer_retrain.utils import *
import types

hours = lambda: time.monotonic()/3.6e3

def pmem():
    print(torch.cuda.memory_allocated())

class TrainerConfig():
    def __init__(self,
        max_grad_norm: float,
        loss: str,
        dloader_workers: int,
        batch_size: int,
        save_name: str,
        wandb_kwargs: dict,
        flags: dict,
        run_id: str,
        TIMEOUT: float,
        PRINT_INTERVAL: int,
        CHECKPOINT_INTERVAL: float,
        SHORT_TRAIN_ITERATIONS: int = None,
        ):
        '''
        All simple python types required to run trainer

        args:
            dloader_workers: passed directly to dataloader objects
            batch_size: passed directly to dataloader objects
            save_name: name of the .pt file where model will be saved once training is complete
            wandb_kwargs: wandb args to pass directly to wandb.init TrainerConfig will
                create separate wandb runs for train and valid splits at initialization.
                
                TrainerConfig will add resume and id fields to kwargs. The proper wandb run
                will be used when Trainer.run_epoch is called
            flags: booleans to configure training. Right now they're just used for debugging
        '''
        self.flags = flags
        
        # Training parameters
        self.max_grad_norm = max_grad_norm
        self.split = 'train' # is always either 'train' or 'valid'
        self.it = 0
        self.loss = loss
        # Performance
        self.dloader_workers = dloader_workers
        self.batch_size = batch_size
        self.TIMEOUT = TIMEOUT
        # Logging, saving
        self.id = run_id
        self.epoch = 0
        self.PRINT_INTERVAL = PRINT_INTERVAL
        self.CHECKPOINT_INTERVAL = CHECKPOINT_INTERVAL
        self.SHORT_TRAIN_ITERATIONS = SHORT_TRAIN_ITERATIONS
        self.checkpoint_path = join(root,'checkpoint',self.id+'.pt')
        self.save_path = join(root,'save',save_name+'.pt')
        self.wandb_kwargs = wandb_kwargs
        wandb_kwargs['resume']='allow'

    def wandb_init(self):
        kwargs = self.wandb_kwargs.copy()
        kwargs['id'] = self.id
        kwargs['notes'] = self.id
        wandb.init(**kwargs)


class Trainer():

    def __init__(self,
                train_data: Dataset,
                validation_data: Dataset,
                training_schedule: TrainingSchedule,
                model: torch.nn.Module,
                config: TrainerConfig,
                optimizer: optim.Optimizer,
                metrics: list[torch.nn.Module],
            ):
        '''
            args:
                train_data: dataset instance of training data. Must contain 'features' and 'targets' fields
                validation_data: dataset instance of validation data. Must contain 'features' and 'targets' fields
                lr_schedule: instance of lr schedule
                model: instance of model to train. Must produce dict output
                    and accept features and targets as args
                config: instance of TrainerConfig. TrainerConfig takes all primative
                    parameters, Trainer itself will take instances of objects used for
                    training
                optimizer: instance of optimizer
                dataloader: if resuming from a checkpoint, dataloader
        '''
        
        # Regardless of whether we loaded the model or not, it will always be initialized on CPU,
        # so compile and load to GPU will always be necessary
        if device() == 'cuda':
            model = model.to(device())
            torch.compile(model)
            for metric in metrics.values():
                metric.to(device())
                torch.compile(metric)

        self.model = model
        self.optimizer = optimizer
        self.training_schedule = training_schedule
        self.train_data = train_data
        self.validation_data = validation_data
        self.config = config
        self.metrics = metrics
        
        self.last_checkpoint_time = hours()
        self.start_time = hours()
        self.timeout = False

    @property
    def runtime(self):
        return hours() - self.start_time

    def set_timeout(self):
        # We want to only set timeout from within run_epoch
        # because we want to know whether run_epoch itself has
        # timeed out, not if the script has been running for more than
        # TIMEOUT hours
        if self.runtime > self.config.TIMEOUT:
            self.timeout = True
        return self.timeout

    def train(self):
        self.config.wandb_init()
        config = self.config
        model = self.model
        training_schedule = self.training_schedule

        config.wandb_init()
        E = None
        while not training_schedule.done:
            try:
                self.run_epoch()
            except Exception as e:
                print("Something went wrong. Checkpointing and aborting")
                E = e
            if self.timeout:
                break
            config.split = 'valid' if config.split == 'train' else 'train'
            config.epoch += 1 if config.split == 'train' else 0
        
        wandb.finish()
        self.checkpoint(force=True)
        if E is not None:
            raise(E)

    def checkpoint(self,force=False):
        config = self.config
        optimizer = self.optimizer
        model = self.model
        training_schedule = self.training_schedule
        metrics = self.metrics

        if not force and not hours() - self.last_checkpoint_time > config.CHECKPOINT_INTERVAL:
            return
        self.last_checkpoint_time = hours()
            
        
        save_dict = {}
        save_dict['config'] = config
        optimizer.zero_grad()

        checkpoint(
            obj = model,
            label = 'model',
            sd = save_dict,
        )
        print(training_schedule.done)
        if training_schedule.done:
                print("done!")
                torch.save(save_dict,config.save_path)
                print("model saved")
                if exists(config.checkpoint_path):
                    os.remove(config.checkpoint_path)
                    print("checkpoint removed")
                return
        checkpoint(
            obj = optimizer,
            label = 'optimizer',
            sd = save_dict,
        )
        checkpoint(
            obj = training_schedule,
            label = 'training_schedule',
            sd = save_dict,
        )

        for m,metric in metrics.items():
            checkpoint(
                obj = metric,
                label = m, # please don't name your metrics something like 'optimizer' lol
                sd = save_dict,
        ) # please don't name your metrics something like 'optimizer' lol
        save_dict['metrics']=list(metrics.keys())
        print(f"saving checkpoint to {config.checkpoint_path}")
        torch.save(save_dict,config.checkpoint_path)

    def run_epoch(self):
        model = self.model
        config = self.config
        split = config.split
        metrics = self.metrics

        dataset = self.train_data if split=='train' else self.validation_data
        dataloader = DataLoader(
            dataset,
            batch_size = config.batch_size,
            num_workers = config.dloader_workers
        )

        optimizer = self.optimizer
        training_schedule = self.training_schedule

        if device()=='cuda':
            scaler = GradScaler()
            forward_context = autocast
        else:
            forward_context = nocontext   

        if split == 'valid':
            # Prevents increased memory usage if your model uses dropout
            grad_context = torch.no_grad
            model.train()
        else:
            grad_context = nocontext
            model.train()

        resume_it = config.it
        last_time = time.monotonic()
        for it, data in enumerate(dataloader):
            if self.set_timeout():
                if it < resume_it:
                    # this is a corner case that I assume will be pretty
                    # nasty to debug if we don't put something here
                    raise TimeoutError("Trainer could not iterate over dataset within alloted time. Aborting")
                return
            self.checkpoint()
            # If debugging flags are enabled, artifically shorten runtime
            if config.SHORT_TRAIN_ITERATIONS is not None and  it > config.SHORT_TRAIN_ITERATIONS:
                break

            # If checkpointed in the middle of an epoch, this will get us 
            # back to where we left off. Basenji is not random
            # access, but loading is trivially
            # inexpensive. This should not come with any performance cost
            if it - 1 < config.it:
                continue
            config.it = it
            if it % config.PRINT_INTERVAL == 0:                
                print(f"({config.epoch}.{it}, {split}): {self.runtime}")

            # Checkpoint every so often so that if something happens at hour 10 we didn't lose
            # half a day of computation
            # 
            # self.checkpoint() calls optimizer.zero_grad, so we don't want to place
            # this in between forward and backward pass like the rest of the logging code
            
            #forward pass
            features = data['features']
            targets = data['targets']
            if not isinstance(features,torch.Tensor):
                assert TypeError(f"data['features'] is not torch.Tensor (got {type(features)})")
            if not isinstance(targets,torch.Tensor):
                assert TypeError(f"data['targets'] is not torch.Tensor (got {type(targets)})")
            features = features.to(device())
            targets = targets.to(device())
            # forward_context is optionally torch.cuda.autocast()
            # grad_context is optionaly torch.no_grad()
            
            with forward_context(), grad_context():
                y_hat = model(features)
                if not isinstance(y_hat, torch.Tensor):
                    raise ValueError(f"model must output torch.Tensor (got {type(y_hat)})")
                results = {}
                for m in self.metrics:
                    results[m] = self.metrics[m](y_hat,targets)
                    try:
                        results[m].item()
                    except:
                        raise ValueError(f"Could not reduce output of {m} to scalar (shape = {results[m].shape})")
            #bookkeeping
            log = {}
            for result in results:
                log[f"{split}/{result}"] = results[result].item()
            log['time'] = time.monotonic()-last_time
            last_time = time.monotonic()
            if split == 'train':
                log['learning rate'] = training_schedule.last_lr
            log['iteration'] = it + config.epoch * len(dataset)
            wandb.log(log)
            #backward pass
            if split == 'train':       
                optimizer.zero_grad()
                loss = results[config.loss]
                if forward_context==nocontext: 
                    loss.backward()
                    if config.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(),config.max_grad_norm)
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if config.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(),config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                training_schedule.step()
                optimizer.zero_grad()
        for m in metrics:
            if hasattr(metrics[m],'reset'):
                metrics[m].reset()
        config.it = 0
        return 'EPOCH COMPLETE'



# python does not allow conditional context managing
# so anywhere where we want to conditionally have no
# context, we sub this in
class nocontext:
    def __enter__(self):
        pass
    def __exit__(self,*args):
        pass       
