# Inference

XXX: this is super-early - please ignore for now - just gathering content at this stage.


## Glossary

- FHE: Fully Homomorphic Encryption
- LPU: Language Processing Unit™
- MPC: Secure Multi-Party Computation
- PPML: Privacy-Preserving Machine Learning
- TTFT: Time to First Token

See [Concepts](#concepts) for more glossary-like entries.

## Key performance metrics

While there are many characteristics an inference server can be judged by - like power usage, efficiency and cost, the most important characteristics are all in the domain on having a smooth user experience. If the user experience is slow and choppy, the user will go to a competitor. Therefore the key needs are:

1. Fast Average Time to First Token (TTFT) - everybody expects any application to start responding ideally faster than 1 second. Therefore the shorter the time the user has to wait before they start receiving the fist tokens the better. This becomes even more important for chatbots which are expected to be interactive.

2. High Token Generation Rate - usually generated tokens per second - this is an indicator of how well the inference server can handle heavy incoming traffic. The higher the rate the more scalable the system that is facing its users, the better the user's experience.

It's important to observe that TTFT w/o a load on a server can be very different from when a server is under a heavy load. If normally the server sends the first token in 1 sec, if the server is already busy processing all the requests it can handle at once and there is a queue, the real TTFT other than for the first few requests, could easily be much much longer. So usually one should measure an average TTFT and report it together with the number of concurrent requests sent during the benchmark.

## Concepts

### Prefill and Decode

When doing inference there are 2 stages:

1. Prefill: as all tokens of the prompt are known - process the full prompt length at once (similar to training) and cache the intermediate states (KV cache). This stage contributes very little latency as even a 1k prompt can be processed really fast, given enough memory.

2. Decode: new tokens generation happens, one new token at a time (regressive approach) based on all the previous tokens (the prompt and any new tokens generated so far). Thus this stage contributes the most to the generation's latency as unlike prefill, decoding can't be parallelized.

### Online vs Offline Inference

When you have users that send queries in real time - this is Online Inference. Examples: Chatbot, search engines. In this case one always runs an inference server and there could be various clients connecting to it.

When you have a file with prompts that you need to run inference on - this is Offline Inference. Examples: benchmark evaluation, synthetic data generation. In this case the inference server is often not needed and the inference is run directly in the same program that sends the query (client and server in one application).

### Batching

Processing the decoding stage one token at a time is extremely accelerator-inefficient. Batching multiple queries together improved the accelerator utilization and enables processing multiple requests at once.

The maximum possible batch size depends on how much memory is left after loading the model weights and filling the KV-cache with intermediate states.

#### Static batching

This is the naive straightforward batching where the first N queries are batched together - the problem here is that if many queries have finished generating they will have to wait for the longest to generate query to complete before they can be returned to the caller - greatly increasing the latency.

#### In-flight batching

In-flight batching is a process where the generation engine removes completed results as soon as they are done and replacing them with new queries, without waiting for the whole batch to complete. So that a sequence in position 0 in the batch could be generating its 10th token, while a sequence in position 1 in the batch could be just starting its first token generation, and position 3 is producing its last token.

This improves the response time, since there is no need for a sequence that already finished not to be returned immediately and there is no need for a new prompt to wait for the next batch to become available. Of course, if all of the compute is fully busy, and there are no new openings in the batch, then some requests will have to wait before the compute will start processing those.

### Structured Text Generation

Also known as Guided Text Generation.

If the model can return its generated output in a specific format, rather than unrestricted format, you don't want the model to hallucinate invalid formats. For example, if you want a model to return a JSON dict, it should do just that.

The way this is accomplished is by using guided text generation. Instead of choosing a generated token with highest probability, the technique uses the next best token with highest probability that fits the next expected token sub-set. To elucidate with an example: if you want the model to generate a JSON list of strings like `["apples", "oranges"]` thus we expect:

```
["string", "string", ..., "string"]
123...
```

The first generated token has to be `[`. If the model got `"`, for example, instead of `[`, as the highest probability, and `[` had a lower probability - we want to pick the one with lower probability so that it'll be `[`.

Then the next generated token has to be `"`. If it's not, search for tokens with lower probabilities until `"` is found and choose that.

The third token has to be a valid string (i.e. not `[` or `"`).

And so on.

Basically, for each next token we need to know a subset of tokens that is allowed and choose

This is a very cool technique. Instead of trying to repair the generated output which is not always possible to match the expected format, we get the model to generate the correct output in the first place.

This technique has several costs:
- it slows down the generation - the more complex the schema it has to adhere to the slower it'll be at generating tokens. From measuring generation speed I found some structured text generation libraries perform much faster than others.
- it may contribute to model hallucination.

There are multiple implementations of this technique, as of this writing 2 popular libraries are:
- https://github.com/outlines-dev/outlines
- https://github.com/noamgat/lm-format-enforcer

You ideally want the implementations that have already integrated into inference frameworks like `vllm` and others.



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



### Privacy-preserving inference

Most companies serving inference will run into a user privacy need. It should be safe for a user to submit a query w/o someone snooping on it. One solution would be an on-premise solution where the client runs the server themselves and then there is no privacy issue, but that most likely is going to expose provider's IP - model's weights and possibly code/algorithms. Therefore, there is a need for a fully encrypted generation - that is the computations are to be performed on client-encrypted data.

The solutions that address this need are called Privacy-Preserving Machine Learning (PPML).

One of the solutions is called Fully [Homomorphic Encryption](https://en.wikipedia.org/wiki/Homomorphic_encryption) (FHE).

Have a look at one such implementation, [concrete-ml](https://github.com/zama-ai/concrete-ml) that rewrites the model to be able to have the client run part of the model themselves, then the intermediary encrypted activations are sent to the server to perform the attention and then sent back to the client. Thus the provider retains part of their IP - and I suppose this part of IP prevents the client from stealing the full IP, since partial weights aren't enough to reconstruct the full model. [This article](https://huggingface.co/blog/encrypted-llm) goes into more details.

There are various other approaches, e.g. this paper: [LLMs Can Understand Encrypted Prompt: Towards Privacy-Computing Friendly Transformers](https://arxiv.org/abs/2305.18396v3) goes into a custom solution based on Secure Multi-Party Computation (MPC) and FHE and has a good reference list.

The problem with current solutions is the huge computational overhead - which greatly impacts the cost and latency. In the future ASIC solutions should address these issues.



## Inference frameworks

I'm actually not sure if I want/should maintain this list - there are so many of them and more emerging all the time and it's impossible to find time to try them all out... not sure what to do here.

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

### SGLang

https://github.com/sgl-project/sglang

### OpenPPL

https://github.com/OpenPPL/ppl.nn

PPLNN, which is short for "PPLNN is a Primitive Library for Neural Network", is a high-performance deep-learning inference engine for efficient AI inferencing.


### LightLLM

https://github.com/ModelTC/lightllm

LightLLM is a Python-based LLM (Large Language Model) inference and serving framework, notable for its lightweight design, easy scalability, and high-speed performance.

### LMDeploy

https://github.com/InternLM/lmdeploy

### MLC-LLM

https://github.com/mlc-ai/mlc-llm

### TGI

https://github.com/huggingface/text-generation-inference


### Orca

[Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) - C++ inference engine based on NVIDIA's `FasterTransformer` as the generation/execution engine (it looks like `FasterTransformer` got folded into [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM).


## Accelerator-specific frameworks

Most obviously support NVIDIA CUDA. Many support AMD ROCm and Intel Gaudi

But there are accelerator specific frameworks:

### Intel Gaudi, MAX, etc.

-  https://github.com/intel/intel-extension-for-transformers




## Inference Chips

### Groq

- [Groq](https://groq.com/)


## Benchmarks

https://github.com/bentoml/llm-bench - benchmarks inference loads (not yet sure if it works only for BentoML)


## Resources

- [A Survey on Efficient Inference for Large Language Models (2024)](https://arxiv.org/abs/2404.14294)
