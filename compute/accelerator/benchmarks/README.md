# Accelerator Benchmarks

## Maximum Achievable Matmul FLOPS Finder

Maximum Achievable Matmul FLOPS (MAMF) Benchmark: [mamf-finder.py](./mamf-finder.py)

For a detailed discussion and the numbers for various accelerators see [Maximum Achievable FLOPS](../#maximum-achievable-flops).

While some accelerator manufacturers publish the theoretical TFLOPS these usually can't be reached. As a result of this when we try to optimize our software we have no realistic performance bar to compare ourselves to. The Model FLOPS Utilization (MFU) metric measures TFLOPS achieved against theoretical TFLOPS. Usually when one scores around 50% MFU it's considered a win. But this gives us no indication how far are we from the real achievable throughput.

This benchmark scans various large shapes of matmul and reports the highest achievable TFLOPS it registered. As transformers training and partially inference workloads are dominated by large matmul operations it's safe to use the best matmul TFLOPS one can measure on each accelerator as a rough estimation that this is the Maximum Achievable Matmul FLOPS (MAMF). Now instead of the previously used MFU, one can use Model Achievable Matmul FLOPS Utilization (MAMFU).

Therefore now you can compare the TFLOPS you measured for your training or inference against a realistic number. As you will now be much closer to 100% it'll be much easier to know when to stop optimizing.

Currently supported high end architectures:
- NVIDIA: V100, A100, H100, ...
- AMD: MI250, MI300X, ...
- Intel Gaudi2+

Fairness notes:
- if you can find a better and more efficient way to detect the best matmul TFLOPS by approaching each new accelerator as a black box, please kindly send a PR with the improvement including the generated log file.
- also if you know that this benchmark should be run under special conditions to show the best results, such as some kernel settings or similar, please submit a PR to add such special instructions. For example, for AMD MI300X I'm being told disabling the numa_balancing is supposed to help.

### Architecture specific notes:

Follow the special setup instructions before running the benchmark to achieve the best results:

**MI300x**:

Turn numa_balancing off for better performance:
```
sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'
```

### Examples of usage

In the ranges below `N` is the reduction dimension so that `(MxN)*(NxK)=(MxK)` and we print the MxNxK shape for the best measured TFLOPS.

Also by default we use 50 warmup and 100 measured iterations for each shape and then fastest result is picked (not the average). You can change the number of iterations via the args `--num_warmup_iterations` and `--num_iterations` correspondingly.

You can specify the data type via `--dtype` argument, it has to be one of the valid `torch` dtypes - e.g., `float16`, `bfloat16`, `float32`, etc. If not specified `bfloat16` is used.

Here we do `torch.mm(MxN,NxK) -> MxK`

1. A quick run (under 1min) - should give around 80-90% of the maximum achievable result - good for a quick try out, but not enough to get a high measurement.

```
./mamf-finder.py --m_range 0 20480 256 --n 4096 --k 4096 --output_file=$(date +"%Y-%m-%d-%H:%M:%S").txt
```

2. A more exhaustive search (will take much longer) - but you can Ctrl-C it when it run long enough and get the best result so far:

```
./mamf-finder.py --m_range 0 5376 256 --n_range 0 5376 256 --k_range 0 5376 256 --output_file=$(date +"%Y-%m-%d-%H:%M:%S").txt
```

3. A super long exhaustive search (can take many days) - but you can Ctrl-C it when it run long enough and get the best result so far:

```
./mamf-finder.py --m_range 0 20480 256 --n_range 0 20480 256 --k_range 0 20480 256 --output_file=$(date +"%Y-%m-%d-%H:%M:%S").txt
```

4. If you want to measure a specific shape that is used by your training, use the exact shape, instead of the range, so let's say you wanted to measure 1024x1024x1024 - you'd run:

```
./mamf-finder.py --m 1024 --n 1024 --k 1024 --output_file=$(date +"%Y-%m-%d-%H:%M:%S").txt
```

5. Accelerator specific range seeking suggestions

But then it appears that different accelerators have different ranges of shapes that lead to best TFLOPS, thus it's difficult to suggest a range that will work well for all of them - instead here are some suggestions based on experiments and suggestions from contributors:

- **A100** + **MI300X**

```
./mamf-finder.py --m_range 0 5376 256 --n_range 0 5376 256 --k_range 0 5376 256 --output_file=$(date +"%Y-%m-%d-%H:%M:%S").txt
```

- **H100**

```
./mamf-finder.py --m_range 0 20480 256 --n_range 0 20480 256 --k_range 0 20480 256 --output_file=$(date +"%Y-%m-%d-%H:%M:%S").txt
```

To understand better which shapes give the highest matmul FLOPS for a particular accelerator, see [Vector and matrix size divisibility](../../../training/performance/README.md#vector-and-matrix-size-divisibility).


### Results

The measurements that I have gathered so far can be found at [Maximum Achievable Matmul FLOPS comparison table](../#maximum-achievable-matmul-flops-comparison-table). When I had access to a particular accelerator I run the benchmarks myself, when I didn't it was the kind contributors who invested their time to get these numbers. So I'm very grateful to [those](../../../contributors.md).
