#!/bin/bash
#SBATCH --job-name=utkg
#SBATCH --ntasks=1
#SBATCH --time=48:00:00

PROJ_DIR=$HOME/WORK/projects/deepsymbolic/code/
PYTHONPATH=$PYTHONPATH:$PROJ_DIR
export PYTHONPATH

python3 run.py

