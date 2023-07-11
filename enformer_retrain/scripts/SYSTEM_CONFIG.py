'''
system configurations. Will be preserved between resubmissions since
they're passed as command-line arguments

WANDB_PROJECT is specified in wandb_project.txt, which is listed
in the .gitignore, in order to prevent 
us from accidentally logging jobs to each other's projects after
doing a git pull
'''

config_dict = {
    'trainer': {
        'TIMEOUT': 11.0, # hours
        'PRINT_INTERVAL': 1000, #iterations
        'CHECKPOINT_INTERVAL': 1.0, # hours
    },

    'trainer_db': {
        'FREQUENT_CHECKPOINT_INTERVAL': 1/3600, # hours
        'QUICK_TIMEOUT': 30/3600, # hours
        'SHORT_TRAIN_ITERATIONS': 50, #iterations
    },
    'qsub_resources': {
        'gpus' : 1,
        'gpu_c': '7.0',
    }
}
