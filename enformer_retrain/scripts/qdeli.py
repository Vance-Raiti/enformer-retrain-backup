import os
import sys
import subprocess
from os.path import join
'''
Delete all jobs matching the provided run id

args:
    1: run id
'''



if len(sys.argv) != 3:
    print("usage: qdeli [user] [run id]")
    exit()
user = sys.argv[1]
run_id = sys.argv[2]

lines = subprocess.check_output(['qstat','-u',user])
lines = str(lines).split('\\n')
if len(lines)<3:
    print(f"User {user} currently has no jobs")
    exit()
script_dir = os.path.dirname(__file__)
log_dir = os.path.join(script_dir,'..','logs')

if not any([run_id in line for line in lines]):
    print(f"No jobs matching {job_id} were found")

for line in lines:
    if run_id not in line:
        continue
    line = line.split(' ')
    job_id = line[0]
    os.system(f"qdel {job_id}")
    with open(join(log_dir,'qdel.log'),'a') as fp:
        fp.write(f'{run_id} ({job_id}) was deleted\n')

