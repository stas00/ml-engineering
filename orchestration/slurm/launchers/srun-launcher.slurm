#!/bin/bash

# this is a 2 node SLURM script for launching srun-based programs
# Important: you will need to adapt setting where you see EDIT in the comments

#SBATCH --job-name=srun-launcher
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8          # EDIT this has to match the number of GPUs per node
#SBATCH --cpus-per-task=10           # EDIT how many cpu cores per task (total-cores/tasks-per-node)
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

# we are preparing for torch.distributed programs so it wants:
# - MASTER_ADDR, MASTER_PORT, WORLD_SIZE - already known before `srun`
# - RANK, LOCAL_RANK - will set at `srun` command
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=6000
export WORLD_SIZE=$SLURM_NPROCS

# srun acts as the launcher in this case, so just `python` is enough.
LAUNCHER="python -u"

# EDIT the path+name of the python script and whatever args it needs
PROGRAM="torch-distributed-gpu-test.py"

export CMD="$LAUNCHER $PROGRAM"

echo $CMD

# EDIT if you want to redirect /tmp to /scratch (some local SSD path) since /tmp is tiny on compute nodes
# export TMPDIR=/scratch

# EDIT: useful for debug if needed
#
# to debug NCCL issues
# export NCCL_DEBUG=INFO
#
# to unravel async errors w/o the correct traceback - potentially makes everything very slower
# export CUDA_LAUNCH_BLOCKING=1
#
# to force crashing on nccl issues like hanging broadcast
# export NCCL_ASYNC_ERROR_HANDLING=1

# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --jobid $SLURM_JOB_ID \
    "

# bash -c is needed for the delayed interpolation of env vars to work
# we want $SLURM_PROCID and $SLURM_LOCALID values that get set at the actual process launch time
srun $SRUN_ARGS bash -c "RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID $CMD" 2>&1 | tee -a $LOG_PATH

echo "END TIME: $(date)"
