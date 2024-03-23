#!/usr/bin/bash

# this helper util performs NUMA node binding which can be used with torchrun, and other launchers
# contributed by https://github.com/yifuwang

# 1. first make it executable:
#
# chmod a+x ./numa-set.sh
#
# 2. launch torchrun and test that it assigns the cores correctly
#
# torchrun --nproc_per_node=8 --no-python ./numa-set.sh \
# python -c "import os; cs=os.sched_getaffinity(0); print(f"{len(cs)} visible cpu cores: {cs}")'
#
# so if your original torchrun launcher looked like:
#
# torchrun --nproc_per_node=8 --nnodes 2 ... train.py
#
# now it'll become:
#
# torchrun --nproc_per_node=8 --nnodes 2 ... --no-python ./numa-set.sh python train.py

# Query the bus ID for device LOCAL_RANK
BUS_ID=$(nvidia-smi --query-gpu=pci.bus_id -i $LOCAL_RANK --format=csv,noheader)
BUS_ID=${BUS_ID,,}

# Find the numa node for device LOCAL_RANK
NODE=$(cat /sys/bus/pci/devices/${BUS_ID:4}/numa_node)

echo "Starting local rank $RANK on NUMA node $NODE"
numactl --cpunodebind=$NODE --membind=$NODE "$@"
