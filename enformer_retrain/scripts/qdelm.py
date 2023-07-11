import os
import sys
import subprocess
from os.path import *
'''
Lists all current qsub jobs and propmts the user
if they would like to delete them

args:
    1 - The name of the user
'''



if len(sys.argv) != 2:
    print("usage: qdelm [user]")
    exit()
user = sys.argv[1]
os.system(f"qstat -u {user}")

lines = subprocess.check_output(['qstat','-u',user])
lines = str(lines).split('\\n')
if len(lines)<3:
    print(f"User {user} currently has no jobs")
    exit()
script_dir = os.path.dirname(__file__)
log_dir = os.path.join(script_dir,'..','logs')
print("For each job displayed, type 'D' to request delete")
for line in lines[2:]:
    line = line.split(' ')
    if len(line) < 3:
        continue
    job_id = line[0]
    name = line[2]
    run_id = name.split('_')[0]
    action = input(f"{run_id}: ")
    if action == 'D':
        os.system(f"python {join(dirname(__file__),'qdeli.py')} {user} {job_id}")

