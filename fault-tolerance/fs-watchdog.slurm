#!/bin/bash
#SBATCH --job-name=fs-watchdog       # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=2:00:00               # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --partition=compil

set -e

echo "START TIME: $(date)"

source $six_ALL_CCFRWORK/start-prod

echo "running partition watchdog"

BIG_SCIENCE_REPO_PATH=$six_ALL_CCFRWORK/code/tr11-176B-ml/bigscience

$BIG_SCIENCE_REPO_PATH/tools/fs-watchdog.py

echo "END TIME: $(date)"
