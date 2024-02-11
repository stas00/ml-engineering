#!/bin/bash
#SBATCH --job-name=tr11-176B-ml      # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1                    # number of nodes
#SBATCH --cpus-per-task=1            # number of cores per task
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=0:30:00               # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --partition=compil

echo "START TIME: $(date)"

variant=main
DATA_OUTPUT_PATH=$six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml
CHECKPOINT_PATH=$DATA_OUTPUT_PATH/checkpoints/$variant
REPO_PATH=$DATA_OUTPUT_PATH/tr11-176B-ml-logs
LOGS_PATH=$REPO_PATH/logs/$variant

MAIN_LOG_FILE=$LOGS_PATH/main_log.txt
BIG_SCIENCE_REPO_PATH=$six_ALL_CCFRWORK/code/tr11-176B-ml/bigscience
WATCH_SLURM_NAME=tr11-176B-ml

$BIG_SCIENCE_REPO_PATH/tools/slurm-status.py --job-name $WATCH_SLURM_NAME 2>&1 | tee -a $MAIN_LOG_FILE

echo "END TIME: $(date)"
