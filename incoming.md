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

# Testing

- integrate the features from testing_utils.py


# From Adam Moody's team at LLNL

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
