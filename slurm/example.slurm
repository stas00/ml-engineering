#!/bin/bash

# this is a 2 node slurm job example, you will most likely need to adapt --cpus-per-task and --partition

#SBATCH --job-name=example-job
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=0:10:00
#SBATCH --exclusive
#SBATCH --partition=xyz-cluster
#SBATCH --output=%x-%j.out


set -x -e

# CHANGE HERE THE CONDA EVN AND ANY STARTUP SCRIPTS
source /path/to/start-xxx-user # if you have something to preload before the job
conda activate stas-xxx        # if you have conda env to activate

echo "START TIME: $(date)"

# CHANGE TO CUMMULATIVELY LOG OUTPUTS
LOG_PATH="main_log.txt"

GPUS_PER_NODE=8
NNODES=$SLURM_NNODES

# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

# OTHER LAUNCHERS CAN BE USED HERE
export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --role `hostname -s`: \
    --tee 3 \
    "

# CHANGE HERE THE SCRIPT AND WHATEVER ARGS IT NEEDS
CMD="\
torch-distributed-gpu-test.py \
"

echo $CMD

# hide duplicated errors using this hack - will be properly fixed in pt-1.12
# export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error.json

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1

# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

# py-spy top -s -i -n -- $LAUNCHER --node_rank $SLURM_PROCID --role $SLURMD_NODENAME: $CMD
clear; srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER --node_rank \$SLURM_PROCID --role \$SLURMD_NODENAME: $CMD" 2>&1 | tee -a $LOG_PATH

echo "END TIME: $(date)"
