# Things to add / integrate

# Performance

https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html


# IO chapter

Fastest IO solution

Parallel store:

- GPFS
- Lustre

Super slow solutions:

- NFS


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
