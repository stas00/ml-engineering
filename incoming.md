# Things to add / integrate

# Performance

https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html


# IO chapter

### IO benchmarks:

https://github.com/argonne-lcf/dlio_benchmark



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

# Network

like nccl-tests, another common set of benchmarks used at HPC sites are the OSU microbenchmarks like osu_lat, osu_bw, and osu_bibw.

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
