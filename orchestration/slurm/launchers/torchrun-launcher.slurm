#!/bin/bash

# this is a 2 node SLURM script using `torchrun` launcher
# you will most likely need to adapt setting where you see EDIT in the comments

#SBATCH --job-name=torchrun-launcher
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per node
#SBATCH --cpus-per-task=96           # EDIT this to how many cpu cores the node has
#SBATCH --gres=gpu:8                 # EDIT this if it's not 8-gpus per node
#SBATCH --time=0:10:00               # EDIT the desired runtime
#SBATCH --exclusive
#SBATCH --partition=xyz-cluster      # EDIT to the desired partition name
#SBATCH --output=%x-%j.out

echo "START TIME: $(date)"

# auto-fail on any errors in this script
set -eo pipefail

# logging script's variables/commands for future debug needs
set -x

# EDIT the conda evn and any startup scripts
# source /path/to/start-xxx-user # if you have something to preload before the job
# conda activate stas-xxx        # if you have conda env to activate


LOG_PATH="main_log.txt"

# EDIT if it's not 8-gpus per node
GPUS_PER_NODE=8
NNODES=$SLURM_NNODES

# define the node 0 hostname:port
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --role `hostname -s`: \
    --tee 3 \
    "

# EDIT the path+name of the python script and whatever args it needs
PROGRAM="\
my_train_script.py \
"

export CMD="$LAUNCHER $PROGRAM"

echo $CMD

# useful for debug if needed
# export NCCL_DEBUG=INFO
# export CUDA_LAUNCH_BLOCKING=1

# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$CMD" 2>&1 | tee -a $LOG_PATH

echo "END TIME: $(date)"
