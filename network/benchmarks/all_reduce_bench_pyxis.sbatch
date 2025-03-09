#!/bin/bash
#SBATCH --job-name=all_reduce_bench_pyxis
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=01:00:00

# Set up environment variables for torchrun
GPUS_PER_NODE=8
NNODES=$SLURM_NNODES
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

srun --container-image=nvcr.io#nvidia/pytorch:25.02-py3 \
     --container-mounts=$PWD:/workspace \
     python -u -m torch.distributed.run \
         --nproc_per_node $GPUS_PER_NODE \
         --nnodes $NNODES \
         --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
         --rdzv_backend c10d \
         --max_restarts 0 \
         --role `hostname -s`':' \
         --tee 3 \
         all_reduce_bench.py
