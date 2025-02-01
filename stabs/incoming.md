# Things to add / integrate

# pdf book notes

ideas from Sam: https://github.com/saforem2: https://github.com/stas00/ml-engineering/pull/17#discussion_r1439912709
https://quarto.org/, https://quarto.org/docs/gallery/, https://kevinheavey.github.io/modern-polars/, https://quarto.org/docs/output-formats/pdf-basics.html

# Performance

https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html

Dirk Groeneveld's Script that checks node pairs for speed https://github.com/allenai/OLMo/commit/f91cebdfa299bf55e815d496c367de8b59881c2e
```
#!/bin/bash

NCCL_LIB_DIR=/var/lib/tcpxo/lib64 source /var/lib/tcpxo/lib64/nccl-env-profile.sh

set -euxo pipefail

HOST_VARS=$(sed 's/ \{1,\}/ -x /g' <<<"${!NCCL*} LD_LIBRARY_PATH")
FIRST_HOST=$(( echo "$1" && echo "$2" ) | sort | head -1)
mpirun \
  --mca btl self,tcp \
  --mca btl_tcp_if_include enp0s12 \
  --mca orte_base_help_aggregate 0 \
  -H $1,$2 \
  -np 2 \
  --bind-to none \
  -npernode 1 \
  -tag-output \
  -x ${HOST_VARS} \
  -x NCCL_NET=FasTrak \
  -x GLOO_SOCKET_IFNAME=enp0s12 \
  -x CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -x OMP_NUM_THREADS=16 \
  bash -c "source ~/venv/OLMo/bin/activate && torchrun --nproc_per_node 8 --nnodes=2 --rdzv-backend=c10d --rdzv-endpoint=$FIRST_HOST ~/OLMo/scripts/augusta/all_reduce_bench.py"
```
and to run:
```
# checking all node pairs for reduce perf
fgrep -hv \# ~/hostfiles/hosts | \
parallel -N2 'echo {} $(./check_node_pair.sh {} 2>&1 | fgrep busbw)'
```


# Storage chapter

### Storage benchmarks:

https://github.com/argonne-lcf/dlio_benchmark


Incoming suggestions from Ross Wightman to integrate:

- I'd try to separate volumes by workload, so keep the 'lots of small files', high churn like environments, code separate from bulk storage like datasets, checkpoints. Possibly even split those too since datasets are largely static and checkpoints are being rotated all the time

- When datasets are on network storage, just like bucket storage, they should consist of large files AND be read as large files (sequentially in large chunks, not mmapped!). Avoid seeking within datasets

- Setups like HF datasets can be deceiving, might look like one big file, but often being mmap'd and the IO read pattern is nuts, like 3-4x more iops than if you'd read them as individual files.
  Mmap loading can be turned off, but if that's the case, for a lot of datasets you move a problem into the DataLoader processes, requiring reading too much data into memory at once. Better awareness of tradeoffs for different use cases, and especially using Iterable streaming when appropriate.

- In a way, bucket storage like s3, via the interface limitations, enforces patterns that are reasonable for storage backends like this. It's ooh, it's mounted as a folder, I can do whatever I want (mmap files, write loads of little ones, delete them all, etc) that's the prob.

- One also cannot expect to treat a distributed filesystem like their local disk. If you separated volumes by workload you'd probably be able to utilize much higher % of the total storage. Don't mix high churn, small files with low churn large files.

- Also, note that once your datasets are optimally friendly for a large, distributed network filesystem, they can usually just be streamed from bucket storage in cloud systems that have that option. So better to move them off the network filesystem in that case.

# Debug

Memory leak Checking

```
cuda-memcheck --leak-check full python program.py
```


Race detection:
```
cuda-memcheck --tool racecheck
```
with extra options:
 --save to save output to a disk
 --print-level to control output

```
cuda-memcheck --tool racecheck --racecheck-report analysis
```

gdb with cuda

```
cuda-gdb
```

- integrate debug_utils.py


# model parallelism

a good table here Scaling equations of each type of parallelism.
https://www.cerebras.net/blog/cerebras-sets-record-for-largest-ai-models-ever-trained-on-single-device#summary


# Network

Make a new benchmark section:

1. nccl-tests
2. `all_reduce_bench.py`
3. https://github.com/deepspeedai/DeepSpeedExamples/tree/master/benchmarks/communication
4. like nccl-tests, another common set of benchmarks used at HPC sites are the OSU microbenchmarks like osu_lat, osu_bw, and osu_bibw.

https://mvapich.cse.ohio-state.edu/benchmarks/

Those are MPI-based benchmarks.  Those can be run using GPUDirect RDMA so you can measure MPI performance between GPUs, either on the same node or between nodes.


## Infiniband

References:
- [Sys Admin Pocket Survival Guide - InfiniBand](https://tin6150.github.io/psg/infiniband.html)


### Diagnostics

Not-IB specific
- `ifconfig` - display the status of the currently active interfaces
- `ip addr show` - display the addresses for every link configured on the system

Display the local Host’s IB device status (3 different views).
- `ibstat`
- `ibstatus`
- `ibv_devinfo`

Scan IB network:
- `ibnetdiscover` - scan topology
- `ibroute` - display the unicast and multicast forwarding tables for the switches
- `ibdiagnet` - IB diagnostic net

Check for network errors:
- `ibcheckerrors` - check if the error counters of a port/node are within predefined thresholds
- `ibchecknet` - perform port/node/errors check on the subnet.

Test IB network configuration:
- `ibcheckport` - perform some basic tests on the specified port
- `ibchecknode` - perform some basic tests on the specified node
- `ibclearcounters` - clear port counters for the InfiniBand subnet

Other checks:
- `iblinkinfo`
- `ibcheck`
- `wwibcheck`
- `ibswitch` - verify that an IB-QNEM is installed in the shelf
- `ibhosts` - list all hosts in the IB network.
`ibswitches` - list all ib switches

Tracing:
- `ibping` - ping/pong between InfiniBand nodes
- `ibsysstat` - obtain basic information for remote nodes (hostname, cpus, memory, utilization)
- `ibswitches` - scan the net or use existing net topology file and list all switches
- `ibhosts` - scan the net or use existing net topology file and list all hosts

Display network topology:
- `iblinkinfo -R`

Use `ifconfig` to discover `IPoIB` networks, e.g. if you get `ib0` device with `inet addr:100.1.1.102`, you can connect to it - e.g. `ping 100.1.1.102`

Find the controller:
`lspci | grep Mellanox`

Print driver configuration (interface name comes from `ifconfig`):
`ethtool -i enP49239s1`

### Performance

`perftest` Package includes:
- `ib_send_bw`
- `ib_send_lat`
- `ib_write_bw`
- `ib_write_lat`
- `ib_read_bw`
- `ib_read_lat`
- `ib_atomic_bw`
- `ib_atomic_lat`

example: `ib_send_bw -a address` - test bandwidth

`qperf` measures bandwidth and latency between two nodes (TCP/IP and RDMA transports)



If the network is much slower than it should be, might have to specify which HCAs to use (`ibv_devinfo` to get HCAs)
```
export NCCL_IB_HCA=mlx5
```

might need to install ib packages on the vms:

```
sudo apt-get install -y automake dh-make git libcap2 libnuma-dev libtool make pkg-config udev curl librdmacm-dev rdma-core \
    libgfortran5 bison chrpath flex graphviz gfortran tk dpatch quilt swig tcl ibverbs-utils infiniband-diags
sudo sed -i -e 's/# OS.EnableRDMA=y/OS.EnableRDMA=y/g' /etc/waagent.conf
```

- Verbs: allow command to be executed on feature-rich IB switch.


# SLURM

repos to explore:
https://github.com/OleHolmNielsen/Slurm_tools


# Testing

- integrate the features from testing_utils.py


# From Adam Moody's team at LLNL


- NUMA affinities

https://github.com/LLNL/mpibind/tree/master/python
mpibind for Python enables the use of the mpibind algorithm in arbitrary Python programs.

- Training hanging detection tool:

This is to expand:
https://github.com/stas00/ml-engineering/tree/master/fault-tolerance#is-job-hanging-watchdog


notes from Adam:

https://github.com/LLNL/STAT - the Stack Trace Analysis Tool
https://hpc.llnl.gov/software/development-environment-software/stat-stack-trace-analysis-tool

https://github.com/grondo/io-watchdog

And you can see how we integrated STAT a ways down on this page:

https://hpc.llnl.gov/software/development-environment-software/stat-stack-trace-analysis-tool

There are some "action" scripts one has to write, which io-watchdog executes when it detects a hang.  The contents of those aren't shown on the page, but I could look that up if you're curious.  The user would create a config file like:

```
search /usr/local/tools/io-watchdog/actions
timeout = 20m
actions = STAT, kill
```

This configured io-watchdog to assume the job was stuck if it saw no output for 20 minutes (from rank 0), and then to run "STAT" to collect a stack trace and run "kill" to scancel the job.  We had a couple others, like one to email the user that io-watchdog detected a hang.  One then launches with:
```
srun --io-watchdog mpi_application
```

a quick demo of SCR.  The python to use it is pretty clean.

Install the SCR library (C + MPI)
https://scr.readthedocs.io/en/v3.0/users/build.html#cmake

Install the scr.py module:
https://github.com/LLNL/scr/tree/develop/python#installing-the-scr-python-module

Example checkpoint in python:
https://github.com/LLNL/scr/blob/1878de8756c2b51882a7cda7b97b142eae4e3995/python/scr_example.py#L64-L105



  396  dmesg | grep -i 'limited by'
  397  sudo dmesg | grep -i 'limited by'
  398  nvidia-smi nvlink -e


GPU VBIOS version might be important when researching issues. Let's add the name and bus id to the query, we get:

```
$ nvidia-smi --query-gpu=gpu_name,gpu_bus_id,vbios_version --format=csv

$ nvidia-smi -q | grep "VBIOS Version"
    VBIOS Version                         : 96.00.89.00.01
    [...]
    VBIOS Version                         : 96.00.89.00.01
```


Check error counters of NVLink links

```
$ nvidia-smi nvlink -e
GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-abcdefab-cdef-abdc-abcd-abababababab)
         Link 0: Replay Errors: 0
         Link 0: Recovery Errors: 0
         Link 0: CRC Errors: 0

         Link 1: Replay Errors: 0
         Link 1: Recovery Errors: 0
         Link 1: CRC Errors: 0

         [...]

         Link 17: Replay Errors: 0
         Link 17: Recovery Errors: 0
         Link 17: CRC Errors: 0
```

Another useful command is:
```
$ nvidia-smi nvlink --status
GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-abcdefab-cdef-abdc-abcd-abababababab)
         Link 0: 26.562 GB/s
         [...]
         Link 17: 26.562 GB/s
```
this one tells you the current speed of each link

Run `nvidia-smi nvlink -h` to discover more features (reporting, resetting counters, etc.).

nvidia-smi --query-remapped-rows=gpu_name,gpu_bus_id,remapped_rows.failure,remapped_rows.pending,\
remapped_rows.correctable,remapped_rows.uncorrectable \
--format=csv gpu_name, gpu_bus_id, remapped_rows.failure,remapped_rows.pending,\
remapped_rows.correctable, remapped_rows.uncorrectable


nvidia-smi --query-remapped-rows=gpu_name,gpu_bus_id,remapped_rows.failure,remapped_rows.pending,remapped_rows.correctable,remapped_rows.uncorrectable --format=csvgpu_name, gpu_bus_id, remapped_rows.failure, remapped_rows.pending, remapped_rows.correctable,remapped_rows.uncorrectable
