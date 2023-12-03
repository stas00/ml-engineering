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

https://github.com/LLNL/STAT - the Stack Trace Analysis Tool
