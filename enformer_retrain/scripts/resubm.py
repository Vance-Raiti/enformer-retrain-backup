import sys
from qsub import run_job
from time import sleep
from os.path import *
from os import listdir

'''
Lists all checkpoints and prompts the user to
resume them. 

args:
    None (ignores flags)
'''

if len(sys.argv) != 1:
    print("resubm takes no arguments")
    sleep(1)

print("for each run listed, type \"R\" to resubmit")
checkpoints = join(dirname(__file__),'..','checkpoint')
for fname in os.listdir(checkpoints):
    run_id, extension = fname.split('.')[0]
    if extension != 'pt':
        print(f'skipping {fname}')
    action = input(f'{run_id}: ')
    if action == 'R':
        run_job(['-p','run_id'])

