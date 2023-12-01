# GPU

XXX: This a very early WIP


As of this writing GPUs are the workhorses of the ML training.

There exist two main ML workloads - training and inference. There is also the finetuning workload which is usually the same as training, unless a much lighter [LORA-style](https://arxiv.org/abs/2106.09685) finetuning is performed which requires significantly fever resources and time than normal finetuning.

In language models during inference the generation is performed in a sequence - one token at a time. So it has to repeat the same `forward` call thousands of times one smallish `matmul` (matrix multiplication or GEMM) at a time. And this can be done on either a GPU or some of the most recent CPUs, that can handle inference quite efficiently.

During training the whole sequence length is processed in one huge `matmul` operation. So if the sequence length is 4k long, the training of the same model will require a compute unit that can handle 4k times more operations than inference and do it fast. GPU excels at this task. In fact the larger the matrices it has to multiply, the more efficient the compute.

The other computational difference is that while both training and inference have to perform the same total amount of `matmul`s in the `forward` pass, in the `backward` pass, which is only done for training, an additional 2x times of `matmul`s is done to calculate the gradients wrt inputs and weights. And an additional `forward` is performed if computation recomputation is used. Therefore the training process requires at 3-4x more `matmul`s than inference.


## Bird's eye view on the high end GPU reality

While this might be changing in the future, unlike the consumer GPU market, as of this writing there aren't that many high end GPUs, and if you rent on the cloud, most providers will have more or less the same few GPUs to offer.

GPUs:
- As of today, ML clouds started transitioning from NVIDIA A100s to H100s and this is going to take some months due to the usual shortage of NVIDIA GPUs.
- AMD's MI250 started popping up here and there, but it's unclear when it'll be easy to access those. From a recent discussion with an AMD representative MI300 is not planned to be in general availability until some time in 2025.
- Intel's Gaudi2 is out there as well, but again, I haven't seen many clouds that offer this GPU.
- And there is Graphcore with their IPU offering. You can try these out in [Paperspace](https://www.paperspace.com/graphcore) through their cloud notebooks.

TPU:
- Google's TPUs are, of course, available but they aren't the most desirable GPUs because you can't only rent them, and the software isn't quite yet easily convertible between GPUs and TPUs, and so many (most?) developers remain in the GPU land, since they don't want to be locked in into a hardware which is a Google monopoly.

Pods and racks:
- And then there are various standalone super-computers like Cerebras' WaferScale Engine (WSE) and dozens of different pod and rack configs that compose the GPUs I have just mentioned with super-fast interconnects.

That's about it as Q4-2023.

## The most important thing to understand

I will make the following statement multiple times in this guide - and that it's not enough to buy/rent the most expensive GPUs and expect a high return on investment (ROI). The two metrics for a high ROI for ML training would be:
1. the speed at which the training will finish, because if the training takes 2-3x longer than planned, your model can become irrelevant before it was released - time is everything in the current super-competitive ML market
2. the total $$ spent to train the model, because if the training takes 2-3x longer than planned, you will end up spending 2-3x times more

Unless the rest of the purchased/rented hardware isn't chosen carefully to match the required workload chances are very high that the GPUs will idle a lot and both time and $$ will be lost. The most critical component is [network](../network), then [io](../io), and the least critical ones are ([CPU](../cpu) and [CPU memory](../cpu-memory).

If the compute is rented one usually doesn't have the freedom to choose - the hardware is either set in stone or some components might be replaceable but with not too many choices. Thus there are times when the chosen cloud provider doesn't provide a sufficiently well matched hardware, in which case it's best to seek out a different provider.

If you purchase your servers then I recommend to perform a very indepth due diligence before buying.

Besides hardware, you, of course, need software that can efficiently deploy the hardware.

We will discuss both the hardware and the software aspects in various chapters of this guide.

## What GPU characteristics do we care for

We will use the NVIDIA A100 spec as a reference point in the following sections.

![nvidia-a100-spec](images/nvidia-a100-spec.png)

[source](https://www.nvidia.com/en-us/data-center/a100/)

### TFLOPS

As mentioned earlier most of the work that ML training and inference do is matrix multiplication. If you remember your algebra matrix multiplication is made of many multiplications followed by summation. Each of these computations can be counted and define how many of these operations can be performed by the chip in a single seconds.

This is one of the key characteristics that the GPUs are judged by. The term TFLOPS defines how many FloatingPointOperations the chip can perform in a second. The more the better. There is a different definition for different data types. For example, here are a few entries for A100:

| Data type              | FLOPS | w/ Sparsity |
| :---                   |   --: |         --: |
| FP32                   |  19.5 |        19.5 |
| Tensor Float 32 (TF32) |   156 |         624 |
| BFLOAT16 Tensor Core   |   312 |             |
| FP16 Tensor Core       |   312 |         624 |
| INT8 Tensor Core       |   624 |        1248 |

footnote: INT8 is measured in TeraOperations as it's not a floating operation.

So you can see that int8 is 2x faster than bf16 which in turn is 2x faster than tf32.

Moreover, the TFLOPs depend on the matrices size as can be seen from this table:

![nvidia-a100-matmul-tflops](nvidia-a100-matmul-tflops.png)

[source](https://developer.nvidia.com/blog/cuda-11-features-revealed/)

As you can see the difference in performance is non-linear due to [the tile and wave quantization effects](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#dim-quantization).

#### Actual peak TFLOPS

The problem with the advertised peak TFLOPS is that they are **very** theoretical and can't be achieved in practice even if all the perfect conditions have been provided. Each GPU has its own realistic TFLOPS which is not advertised and there are anecdotal community reports that do their best to find the actual best value, but I'm yet to find any official reports.

If you find solid reports (papers?) showing the actual TFLOPS one can expect from one or more of the high end GPUs discussed in this chapter please kindly submit a PR with this information. The key is to have a reference to a source that the reader can validate the proposed information with.

To provide a numerical sense to what I'm talking about is let's take A100 with its 312 TFLOPS bf16 peak performance in the specs of this card. Until the invent of FlashAttention it was known that 150TFLOPS was close to the highest one could get for half precision mixed precision, with FlashAttention, it's around 180TFLOPS. This is, of course, measured for training LLMs where the network and IO are involved which create additional overheads. So here the peak performance probably lays somewhere between 200 and 300 TFLOPS.

I think it should be possible to calculate the actual peak TFLOPS by doing a perfect max-size for the given gpu perfectly aligned matrices `matmul` measured on a single GPU.

XXX: write a small program to do exactly dynamically figuring out the perfect shapes based on [the tile and wave quantization effects](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#dim-quantization) and max sizes (how?) so that the benchmark isn't hardcoded to a particular GPU.


### GPU Memory size and speed

Typically the more memory the better. At any given time usually most of the model weights aren't being used as they wait for their turn to be processed and thus large memory allows more of the model to be on the GPU memory and immediately available for access and update. When there is not enough memory, sometimes the model has to be split across multiple GPUs, or offloaded to CPU and/or disk.

Current high end GPUs (some aren't GA yet):
- NVIDIA GPUs A100 and H100 have 80GB of HBM, while the dual-gpu H100 NVL has 188GB
- AMD MI250 - 128GB, MI300 - 196GB

Memory speed is of course very important since if it's not fast enough than the GPU compute ends up idling waiting for the data to be copied to and from the memory.

The GPUs use [High Bandwidth Memory](https://en.wikipedia.org/wiki/High_Bandwidth_Memory) (HBM) which is a 3D version of SDRAM memory. For example, A100-SXM comes with HBM2 at 1.6TB/sec, and H100-SXM comes with HBM3 at 3.35TB/s.

### Heat

This is of interest when you buy your own hardware, when you rent on the cloud the provider hopefully takes care of adequate cooling.

The only important practical understanding for heat is that if the GPUs aren't kept cool they will throttle their compute clock and slow everything down (and could even crash sometimes, albeit throttling is supposed to prevent that).


## High end GPUs for LLM/VLM workloads

### Cloud and in-house GPUs

Most common GPUs that can be either rented on compute clouds or purchased:

NVIVDIA:
- [A100](https://www.nvidia.com/en-us/data-center/a100/#specifications) - huge availability but already getting outdated.
- [H100](https://www.nvidia.com/en-us/data-center/h100) - 2-3x faster than A100 (half precision), 6x faster for fp8, slowly emerging on all major clouds.
- [GH200](https://www.nvidia.com/en-us/data-center/grace-hopper-superchip/) - 2 chips on one card - (1) H100 w/ 96GB HBM3 or 144GB HBM3e + (2) Grace CPU w/ 624GB RAM - availability is unknown.

AMD:
- [MI250](https://www.amd.com/en/products/server-accelerators/instinct-mi250) ~= A100 - very few clouds have them
- MI300 ~= H100 - don’t expect until late-2024 or even 2025 to be GA

Intel:
- [Gaudi2](https://habana.ai/products/gaudi2/) ~= H100 - Currently there is a very low availability on [cloud.google.com](https://cloud.google.com) with a long waiting list which supposedly should be reduced in Q1-2024. AWS has the older Gaudi1 via [DL1 instances](https://aws.amazon.com/ec2/instance-types/dl1/).

Graphcore:
- [IPU](https://www.graphcore.ai/products/ipu) - available via [Paperspace](https://www.paperspace.com/graphcore)

### In-house GPU Clusters

Cerebras:
- [clusters](https://www.cerebras.net/product-cluster/)
- [systems](https://www.cerebras.net/product-system/)
based on WaferScale Engine (WSE).

### Cloud only solutions

These can be only used via clouds:

Google
- [TPUs](https://cloud.google.com/tpu) - lock-in, can't switch to another vendor like NVIDIA -> AMD

Cerebras:
- [Cloud](https://www.cerebras.net/product-cloud/)




## GPUs in detail

### NVIDIA

Abbreviations:

- CUDA: Compute Unified Device Architecture (proprietary to NVIDIA)

NVIDIA-specific key GPU characteristics:
- CUDA Cores - similar to CPU cores, but unlike CPUs that typically have 10-100 powerful cores, CUDA Cores are weaker and come in thousands and allow to perform massive general purpose computations (parallelization). Like CPU cores CUDA Cores perform a single operation in each clock cycle.
- Tensor Cores - special compute units that are designed specifically to perform fast multiplication and addition operations like matrix multiplication. These perform multiple operations in each clock cycle. They can execute extremely fast computations on low or mixed precision data types with some loss (fp16, bf16, tf32, fp8, etc.). These cores are specifically designed for ML workloads.
- Streaming Multiprocessors (SM) are clusters of CUDA Cores, Tensor Cores and other components.

For example, A100-80GB has:

- 6912 CUDA Cores
- 432 Tensor Cores (Gen 3)
- 108 Streaming Multiprocessors (SM)


### AMD

AMD-specific key GPU characteristics:
- Stream Processors - are similar in functionality to CUDA Cores - that is these are the parallel computation units. But they aren't the same, so one can't compare 2 gpus by just comparing the number of CUDA Cores vs the number of Stream Processors.
- Compute Units - are clusters of Stream Processors and other components

for example, AMD MI250 has:
- 13,312 Stream Processors
- 208 Compute Units


### Intel Gaudi2

[Architecture](https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Architecture.html)

- 24x 100 Gigabit Ethernet (RoCEv2) integrated on chip - 21 of which are used for intra-node and 3 for inter-node (so `21*8=168` cards for intra-node (262.5GBps per GPU), and `3*8=24` cards for inter-node (2.4Tbps between nodes)
- 96GB HBM2E memory on board w/2.45 TBps bandwidth per chip, for a total of 768GB per node

A server/node is built from 8 GPUs, which can then be expanded with racks of those servers.

There are no official TFLOPS information published (and from talking to an Intel representative they have no intention to publish any.) They publish the [following benchmarks](https://developer.habana.ai/resources/habana-models-performance/ but I'm not sure how these can be used to compare this compute to other providers.

Comparison: supposedly Gaudi2 competes with NVIDIA H100





## API


### NVIDIA

uses CUDA



### AMD

uses ROCm



### Intel Gaudi

The API is via [Habana SynapseAI® SDK](https://habana.ai/training-software/) which supports PyTorch and TensorFlow.

Useful integrations:
- [HF Optimum Habana](https://github.com/huggingface/optimum-habana) which also includes - [DeepSpeed](https://github.com/microsoft/DeepSpeed) integration.







## Apples-to-apples Comparison

It's very difficult to compare specs of different offerings since marketing tricks get deployed pretty much by all competitors so that one can't compare 2 sets of specs and know the actual difference.

- [MLPerf via MLCommons](https://mlcommons.org/en/) publishes various hardware benchmarks that measure training, inference, storage and other tasks' performance. For example, here is the most recent as of this writing [training v3.0](https://mlcommons.org/en/training-normal-30/) and [inference v3.1](https://mlcommons.org/en/inference-datacenter-31/) results.

   Except I have no idea how to make use of it - it's close to impossible to make sense of or control the view. This is a great intention lost in over-engineering and not thinking about how the user will benefit from it, IMHO. For example, I don't care about CV data, I only want to quickly see the LLM rows, but I can't do it. And then the comparisons are still not apples to apples so how can you possibly make sense of which hardware is better I don't know.
