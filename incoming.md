# Things to add / integrate

# pdf book notes

ideas from Sam: https://github.com/saforem2: https://github.com/stas00/ml-engineering/pull/17#discussion_r1439912709
https://quarto.org/, https://quarto.org/docs/gallery/, https://kevinheavey.github.io/modern-polars/, https://quarto.org/docs/output-formats/pdf-basics.html

# Performance

https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html


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
3. https://github.com/microsoft/DeepSpeedExamples/tree/master/benchmarks/communication
4. like nccl-tests, another common set of benchmarks used at HPC sites are the OSU microbenchmarks like osu_lat, osu_bw, and osu_bibw.

https://mvapich.cse.ohio-state.edu/benchmarks/

Those are MPI-based benchmarks.  Those can be run using GPUDirect RDMA so you can measure MPI performance between GPUs, either on the same node or between nodes.




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
