#!/bin/bash -l

# job.sh 
# function:
#   main qsub script. Simply cd's into the specified path,
#   loads the venv, and runs train.py
#
# args:
#   $1: path of enformer_retrain
#   $2..$end: positional and keyword arguments for
#       train.py

# Request 4 CPUs
#$ -pe omp 4

#specify a project
#$ -P aclab

# 1: script_dir
# 2: qsub count
# 3: run_id
# [4:]: kwargs
cd $1
source loadenv.sh
cd pipeline
python train.py $@
