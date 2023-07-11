# /enformer_retrain

Repository to aid in efforts to replicate results demonstrated by Enformer model (https://www.nature.com/articles/s41592-021-01252-x) using a single 16 GB GPU. This program is designed especially with respect to the features and limitations of the Boston University Shared Computing Cluster.



<ins>**Training loop**


To submit jobs, create a `wandb_project.txt` file in the base directory specifying your wandb project. If you'd like to submit a single job, edit `kwargs` in `/scripts/qsub.py` and execute it. If you'd like to submit a batch of jobs, edit `kwargs` in `/scripts/supersub.py` and execute it. The script actually executed by compute node is `job.sh`. This will call `train.py`, which uses modules declared in `/modules` along with the specified `kwargs` to construct a set of pytorch objects before calling into `trainer.py` to perform the training loop.

To prevent lengthy jobs from timing out, `trainer.py` will automatically exit  if the train length exceeds the number of hours specified in `SYSTEM_CONFIG.py`. In this case, `train.py` will checkpoint the model in `/checkpoint` using a standardized routine declared in `utils.py` and return a unique id for the model. `job.sh` will pass the id to `resub.py`, which will submit a new job to resume training.

When the model completes, its state dict, initialization kwargs, and `TrainerConfig` will be saved to `/save`. The model can be restored using `ld` in `/modules/utils.py`.

<ins>**Adding To Existing Modules (Optimizers, Learning Rate Schedules, Metrics, Models)**

`/modules` currently contains 4 files for declaring pytorch objects to be used in training: `lr_schedules.py`, `metrics.py`, `models.py`, and `optimizers.py`. Each contains a dictionary that associates a string with a different pytorch class. In `train.py`, the associated keyword argument will be used to index into each dictionary to access the module needed for the current job. To insert a new option, simply define it in the appropriate file, making sure that it inherits from the appropriate parent class (specified below), and append it to the dictionary at the end of the file. 

* **Model**: torch.nn.Module
* **Optimizer**: torch.optim.Optimizer
* **Learning Rate Schedule**: torch.optim.lr_schedule
* **Metric**: torch.nn

Note that for any model defined in `model.py`, model.forward must produce a single `torch.Tensor` as output, and any metric defined in `metric.py` must produce a single `torch.Tesnor` scalar as output.


<ins>**Debug Options**

`/scripts/qsub.py` submits a single job. `/scripts/supersub.py` submits a job for every possible combination of kwargs specified. Both have the following command-line debugging options:
- <ins>\-n (no_qsub)</ins>: execute `source job.sh` instead of `qsub job.sh`. 
- <ins>\-t (short_epoch)</ins>: causes all epochs in `trainer.py` to be SHORT_EPOCH_ITERATIONS iterations long (specified at top of `trainer.py`). 
- <ins>\-q (quick_timeout)</ins>: causes the trainer to timeout after QUICK hours (also specified at top of `trainer.py`)
- <ins>\-w (no_wandb)</ins>: will replace wandb with NoOpModule in `train.py` and `trainer.py`, causing nothing to be logged to wandb

Additionally, 'chop' is a simple (1 conv layer) architecture provided in `/modules/models.py` for pipeline testing.

Debug options will be passed to all resubmissions of the job (e.g. if you use `-n`, then when the model times out it will be resubmitted via `source` instead of `qsub`)
