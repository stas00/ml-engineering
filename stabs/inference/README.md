# Inference

XXX: this is super-early - please ignore for now - just gathering content at this stage.


## Glossary

- LPU: Language Processing Unit™

## Concepts

### Prefill and Decode

When doing inference there are 2 stages:

1. Prefill: as all tokens of the prompt are known - process the full prompt length at once (similar to training) and cache the intermediate states (KV cache). This stage contributes very little latency as even a 1k prompt can be processed really fast, given enough memory.

2. Decode: new tokens generation happens, one new token at a time (regressive approach) based on all the previous tokens (the prompt and any new tokens generated so far). Thus this stage contributes the most to the generation's latency as unlike prefill, decoding can't be parallelized.



### Batching

Processing the decoding stage one token at a time is extremely accelerator-inefficient. Batching multiple queries together improved the accelerator utilization and enables processing multiple requests at once.

The maximum possible batch size depends on how much memory is left after loading the model weights and filling the KV-cache with intermediate states.

#### Static batching

This is the naive straightforward batching where the first N queries are batched together - the problem here is that if many queries have finished generating they will have to wait for the longest to generate query to complete before they can be returned to the caller - greatly increasing the latency.

#### In-flight batching

In-flight batching is a process where the generation engine removes completed results as soon as they are done and replacing them with new queries, without waiting for the whole batch to complete. So that a sequence in position 0 in the batch could be generating its 10th token, while a sequence in position 1 in the batch could be just starting its first token generation, and position 3 is producing its last token.

This improves the response time, since there is no need for a sequence that already finished not to be returned immediately and there is no need for a new prompt to wait for the next batch to become available. Of course, if all of the compute is fully busy, and there are no new openings in the batch, then some requests will have to wait before the compute will start processing those.


### Speculative inference

Because it's very slow to generate tokens one a time, sometimes it is possible to cheat and speed things up by using a much smaller and faster draft model. So for example, your normal inference uses Llama-70B which would be quite slow, but we could use Llama-7b as a draft model and then we could verify if the prediction is correct but doing it at once for all tokens.

Example: let's take a prompt `I'm turnin', turnin', turnin', turnin', turnin' around and all that I can see is just` and now:

1. use Llama-7b to predict `another lemon tree` auto-regressively, in 3 steps, but much faster than Llama-70b.
2. now use Llama-70b to run a batch of 3 prompts:

```
[...I can see is just]
[...I can see is just another]
[...I can see is just another lemon]
```
I shortened the full prompt for the sake of the demo with `...` - it should be there for real. And I'm pretending that each token is a full word here.

And now in a single step Llama-70B generates:

```
[...I can see is just] another
[...I can see is just another] lemon
[...I can see is just another lemon] tree
```

Now there could be multiple outcomes:
- if everything matches - in 3 short and 1 long step we generated the final result, instead of using 3 long steps.
- if only `another lemon` matched - we might still better off if it saved time.
- if nothing or little matched we wasted a bit of time.

Obviously, if instead of 3 tokens we had more tokens the savings are likely to be bigger.

Also, don't miss the fact that we did the same amount of compute here and then some, as compared to doing this generation with the large model normally, but the latency of this approach can be much better - so the user on average should get a better response time from your application using it - if the draft model is much smaller and still produces good predictions.


### Key-value caching

It'd be very expensive to recalculate all the previous KV-values before each new token is generated and thus they are cached in accelerator's memory. Newly computed KV-values are appended to the existing cache.

![computation process with caching inference](images/infer-kv-cache.png)

([source](https://developer.nvidia.com/blog/accelerated-inference-for-large-transformer-models-using-nvidia-fastertransformer-and-nvidia-triton-inference-server/))

Some caches are per model, others are per layer.


### Memory requirements

1. Model weights - `model_size_in_Billion_parameters * dtype_size_in_bytes` - e.g. fp16/bf16 is 2 bytes, fp32 is 4 bytes - so a 70B param model in bf16 needs `70*2=140` GB of accelerator memory.
2. Activation memory - this is the processing temp memory which would depend on batch size and sequence length
3. KV Cache of attention tensors - the cache size per token is usually `2*hidden_size*num_hidden_layers*dtype_size_in_bytes`, where 2 stands for K and V caches. For example for LLama2-70B in bf16 it's `2*8192*80*2` => 2.6MB per token (`hidden_size=8192` and `num_hidden_layers=80`). And for 1024 tokens and a batch size of 16, that would add up to 42.5GB.


### Model parallelism

When a model can't fit onto a single accelerator or when it's more efficient to split the model across multiple accelerators even if it does fit but barely, the same [Model Parallelism techniques](../training/model-parallelism) from training apply to inference.



## Inference frameworks


### DeepSpeed-FastGen

[DeepSpeed-FastGen](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen) is an inference system framework for large language models (LLMs) from the DeepSpeed team.

[Updates](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen/2024-01-19).

paper: [DeepSpeed-FastGen: High-throughput Text Generation for LLMs via MII and DeepSpeed-Inference](https://arxiv.org/abs/2401.08671)


#### Dynamic SplitFuse

[Dynamic SplitFuse](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen#b-dynamic-splitfuse-) leverages dynamic prompt and generation decomposition and unification to improve continuous [batching](#batching) and system throughput.





### vLLM

[vLLM](https://github.com/vllm-project/vllm)





### TensorRT-LLM

[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) (also integrated what used to be `FasterTransformer`)




### TGI




### Orca

[Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) - C++ inference engine based on NVIDIA's `FasterTransformer` as the generation/execution engine (it looks like `FasterTransformer` got folded into [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM).


## Inference Chips

### Groq

- [Groq](https://groq.com/)
