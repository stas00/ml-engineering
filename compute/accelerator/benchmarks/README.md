# Accelerator Benchmarks

## Maximum Achievable Matmul FLOPS Finder

Maximum Achievable Matmul FLOPS (MAMF) Benchmark: [mamf-finder.py](./mamf-finder.py)

For a detailed discussion and the numbers for various accelerators see [Maximum Achievable FLOPS](../#maximum-achievable-flops).

While some accelerator manufacturers publish the theoretical TFLOPS these usually can't be reached. As a result of this when we try to optimize our software we have no realistic performance bar to compare ourselves to. The Model FLOPS Utilization (MFU) metric measures TFLOPS achieved against theoretical TFLOPS. Usually when one scores around 50% MFU it's considered a win. But this gives us no indication how far are we from the real achievable throughput.

This benchmark scans various large shapes of matmul and reports the highest achievable TFLOPS it registered. As transformers training and partially inference workloads are dominated by large matmul operations it's safe to use the best matmul TFLOPS one can measure on each accelerator as a rough estimation that this is the Maximum Achievable Matmul FLOPS (MAMF). Now instead of the previously used MFU, one can use Model Achievable Matmul FLOPS Utilization (MAMFU).

Therefore now you can compare the TFLOPS you measured for your training or inference against a realistic number. As you will now be much closer to 100% it'll be much easier to know when to stop optimizing.

Currently supported high end architectures:
- NVIDIA: V100, A100, H100, ...
- AMD: MI250, MI300X, MI325X, ...
- Intel Gaudi2/3

Fairness notes:
- if you can find a better and more efficient way to detect the best matmul TFLOPS by approaching each new accelerator as a black box, please kindly send a PR with the improvement including the generated log file.
- also if you know that this benchmark should be run under special conditions to show the best results, such as some kernel settings or similar, please submit a PR to add such special instructions. For example, for AMD MI300X I'm being told disabling the numa_balancing is supposed to help.

### Architecture specific notes:

Follow the special setup instructions before running the benchmark to achieve the best results:

**MI300x, MI325X, etc.**:

1. Turn numa_balancing off for better performance:
```
sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'
```
2. Enable:
```
export PYTORCH_TUNABLEOP_ENABLED=1
```
This will make the first iteration very slow, while it's searching for the best GEMM algorithm in the BLAS libraries for each `matmul` shape it encounters, but subsequent operations are likely to be significantly faster than the baseline. See [Accelerating models on ROCm using PyTorch TunableOp](https://rocm.blogs.amd.com/artificial-intelligence/pytorch-tunableop/README.html) (requires `torch>=2.3`) [doc](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/tunable/README.md).

**Intel dGPUs (A770, A750, B580, etc.)**
- Follow Intel Extension for Pytorch [installation steps](https://pytorch-extension.intel.com/installation?platform=gpu)

### Examples of usage

In the ranges below `K` is the reduction dimension so that `(MxK)*(KxN)=(MxN)` and we print the MxKxN shape for the best measured TFLOPS.

Also by default we use 50 warmup and 100 measured iterations for each shape and then fastest result is picked (not the average). You can change the number of iterations via the args `--num_warmup_iterations` and `--num_iterations` correspondingly.

You can specify the data type via `--dtype` argument, it has to be one of the valid `torch` dtypes - e.g., `float8_e4m3fn`, `float16`, `bfloat16`, `float32`, etc. If not specified `bfloat16` is used.

Here we do `torch.mm(MxK,KxN) -> MxN`

1. A quick run (under 1min) - should give around 80-90% of the maximum achievable result - good for a quick try out, but not enough to get a high measurement.

```
./mamf-finder.py --m_range 0 20480 256 --n 4096 --k 4096 --output_file=$(date +"%Y-%m-%d-%H:%M:%S").txt
```

2. A more exhaustive search (15-30min) - but you can Ctrl-C it when it run long enough and get the best result so far:

```
./mamf-finder.py --m_range 0 16384 1024 --n_range 0 16384 1024 --k_range 0 16384 1024  --output_file=$(date +"%Y-%m-%d-%H:%M:%S").txt
```

Feel free to make the steps smaller from 1024 to 512 or 256 - but it'd 8x or 64x the run time correspondingly. 1k steps should cover the different shape ranges well and fast.

3. A super long exhaustive search (may take many hours/days) - but you can Ctrl-C it when it run long enough and get the best result so far:

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




## How to benchmark accelerators

### CUDA benchmakrs

There are a few excellent detailed write ups on how to perform CUDA benchmarks:

1. [How to Accurately Time CUDA Kernels in Pytorch](https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch)
2. [How to Benchmark Code on CUDA Devices?](https://salykova.github.io/sgemm-gpu#2-how-to-benchmark-code-on-cuda-devices) - this one is different from (1) in that it suggests to set both GPU and Memory clocks, whereas (1) only locks the GPU clock.

You can see these instructions applied in [mamf-finder.py](./mamf-finder.py) (other than clock locking)

Here are some excellent related reads:

- Horace's [Strangely, Matrix Multiplications on GPUs Run Faster When Given "Predictable" Data](https://www.thonking.ai/p/strangely-matrix-multiplications?utm_source=substack&publication_id=1781836&post_id=142508107) shows how benchmarking can be over-reporting if one uses a not normally distributed data and how power impacts performance.
