# Inference

XXX: this chapter is under construction - some sections are complete, some are still starting out

## Glossary

- FHE: Fully Homomorphic Encryption
- LPU: Language Processing Unit™
- MPC: Secure Multi-Party Computation
- PPML: Privacy-Preserving Machine Learning
- TTFT: Time to First Token
- TPOT: Time Per Output Token

See [Concepts](#concepts) for more glossary-like entries.

## Concepts

### Prefill and Decode

When doing inference there are 2 stages:

#### Prefill

Prefill: as all tokens of the prompt are known - process the full prompt length at once (similar to training) and cache the intermediate states (KV cache). This stage contributes very little latency as even a 1k prompt can be processed really fast, given enough memory.

#### Decode

Decode: new tokens generation happens, one new token at a time (regressive approach) based on all the previous tokens (the prompt and any new tokens generated so far). Thus this stage contributes the most to the generation's latency as unlike prefill, decoding can't be parallelized.

### Online vs Offline Inference

When you have users that send queries in real time - this is Online Inference. Examples: Chatbot, search engines. In this case one always runs an inference server and there could be various clients connecting to it.

When you have a file with prompts that you need to run inference on - this is Offline Inference. Examples: benchmark evaluation, synthetic data generation. In this case the inference server is often not needed and the inference is run directly in the same program that sends the query (client and server in one application).

### Batching

Processing the decoding stage one token at a time is extremely accelerator-inefficient. Batching multiple queries together improved the accelerator utilization and enables processing multiple requests at once.

The maximum possible batch size depends on how much memory is left after loading the model weights and filling the KV-cache with intermediate states.

#### Static batching

This is the naive straightforward batching where the first N queries are batched together - the problem here is that if many queries have finished generating they will have to wait for the longest to generate query to complete before they can be returned to the caller - greatly increasing the latency.

#### Continuous Batching or In-flight batching

Continuous Batching or In-flight batching is a process where the generation engine removes completed results as soon as they are done and replacing them with new queries, without waiting for the whole batch to complete. So that a sequence in position 0 in the batch could be generating its 10th token, while a sequence in position 1 in the batch could be just starting its first token generation, and position 3 is producing its last token.

This improves the response time, since there is no need for a sequence that already finished not to be returned immediately and there is no need for a new prompt to wait for the next batch to become available. Of course, if all of the compute is fully busy, and there are no new openings in the batch, then some requests will have to wait before the compute will start processing those.


### Paged Attention

Paged Attention is very popular with inference servers as it allows for a very efficient accelerator memory utilization, by the virtue of approaching the accelerator memory like the OS memory using paging, which allowed dynamic memory allocation and prevents memory fragmentation.


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

There are multiple implementations of this technique, as of this writing the two popular libraries are:
- https://github.com/outlines-dev/outlines
- https://github.com/noamgat/lm-format-enforcer

You ideally want the implementations that have already been integrated into inference frameworks like `vllm` and others.



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


## Key inference performance metrics

There are two ways to look at performance metrics, the usual system metrics of latency and throughput, and the user-experience metrics: Time To First Token (TTFT) and Time Per Output Token (TPOT). Let's look at both pairs.

### System performance metrics

#### Latency

**Latency is the time it took to receive the complete response since a request was sent**.

This includes the time to:
1. receive the request
2. pre-process the prompt (the prefill stage)
3. generate the new tokens of the response (the decoding stage)
4. send the response back to the client.

The time to receive the request and send the response is mostly the same with a small variation due to the differences in the length of the prompt and the generated response. These length variations should have a negligible impact to the total time.

The prefill stage processes all the prompt's tokens in parallel so here as well the variations in the length of the prompt shouldn't make too much of a difference, albeit longer prompts will consume more accelerator memory and impact the total throughput.

The decoding stage is the one most impacted by the length of the generated response since each new token is generated as a separate step. Here the longer the response the longer the decoding stage will be.

If the server doesn't have enough capacity to process all current requests at once and has to queue some of them, then the wait time in the queue extends the latency by that time.

footnote: if you think of car traffic on the road, latency is the time it takes one to drive from point A to point B (e.g. home to office), including the speed limitations due to traffic lights, jams and legal limits.

#### Throughput

Throughput measures the ability of an inference server to process many requests in parallel and batch requests efficiently.

The definition of throughput could be defined by how many requests can be served concurrently, but since some requests get served much faster than others, so that several short requests could be served during the a single long request it makes sense to count the total rate of tokens generated across the system.

Thus a more common definition of **inference throughput is total tokens generated per second across the whole system**.

footnote: if you think of car traffic on the road, throughput is how many cars can move through a given road at any given time. The more lanes the road has and the higher the speed limit the higher the throughput of that road. But clearly some vehicles are short and some are long, so some sort of normalization is needed. For example, ferries calculate how many meters or feet of vehicles they can fit it and thus long vehicles pay more than short ones.



### User experience metrics

While there are many characteristics an inference server can be judged by - like power usage, efficiency and cost, one could say that since the systems interface humans - the most important characteristics are all in the domain on having a smooth user experience. If the user experience is slow and choppy, the user will go to a competitor. Therefore the key needs are:

#### Time To First Token

**Time To First Token (TTFT) is defined as the time that passed since the user hit the Submit button (or Enter) and the moment they have received a first word or a part of the word in return**.

A very low Time To First Token (TTFT) is wanted. These days users are conditioned to expect from any application to start responding ideally faster than 1 second. Therefore the shorter the time the user has to wait before they start receiving the fist tokens the better. This becomes even more important for chatbots which are expected to be interactive. The length of TTFT is impacted by many elements, the key ones being the computation of the [prefill stage](#prefill) (pre-processing the prompt) and whether the request got its processing immediately upon user request received or whether it had to wait in the queue.

It's important to observe that TTFT w/o a load on a server can be very different from when a server is under a heavy load. If normally the server sends the first token in 1 sec, if the server is already busy processing all the requests it can handle at once and there is a queue, the effective TTFT other than for the first few requests, could easily be much much longer. So usually one should measure an average TTFT and report it together with the number of concurrent requests sent during the benchmark.

This is a non-trivial metric since depending on the prompt size the time will vary, so ideally you'd want to normalize it to the number of tokens in the prompt.

#### Time Per Output Token

Time Per Output Token (TPOT) is a per user metric. It measures how long does it take for a new token to be generated for a given user.

A relatively low Time Per Output Token (TPOT) is desired, but it doesn't have to be too high. This time ideally should be close to the reading speed of the human who sent the request. So for example if you serve first graders the TPOT can be quite low, but the more educated the person is the faster TPOT should be to achieve a smooth reading experience.

According to wiki there are [3 types of reading](https://en.wikipedia.org/wiki/Speed_reading#Types_of_reading) and the reading speed is measured in words per minute (WPM).

The average tokens per word can vary from tokenizer to tokenizer, primarily depending on their vocab size and the language(s). Here let's consider an English tokenizer with about 1.5 tokens per word. Now we can convert words per minute (WPM) to tokens per minute (TPM).

And now we just need to divide by 60 to get Tokens Per Second (TPS) and invert to get time per output token (TPOT)

So `TPOT = 60 / (WPM*1.5)` in seconds

| Reader   | WPM |  TPM |   TPS |  TPOT |
| :-----   | --: | ---: | ----: | ----: |
| Subvocal | 250 |  375 |  6.25 |  0.16 |
| Auditory | 450 |  675 | 11.25 | 0.089 |
| Visual   | 700 | 1050 | 18.75 | 0.057 |

Remember to change the 1.5 co-efficient to the actual word to tokens average ratio of your tokenizer. For example, as of this writing OpenAI ChatGPT's with a 50k vocab is reported to be about 1.3 tokens per word, while many other LLMs have 30k vocabs, which lead to a higher tokens per words ratio.

As you can see TPOT is an awkward value to track and think of in one's head, so **once you know your targeted TPOT it's better to convert it to Tokens Per Seconds (TPS) and track that instead**.

Therefore in this example if your system can generate a sustainable 20 tokens per second per request your clients will be satisfied since that system will be able to keep up even with the super-fast readers at 700 words per minute.

And there, of course, will be users who would prefer to wait till the generation is complete before they would start reading the response. In which case faster is better.

Depending on the type of generation, the following is likely to apply:
1. Image - all-at-once
2. Text - as fast as user's reading speed or all-at-once if they prefer not to have moving parts before they start reading
3. Audio - as fast as user's listening speed
4. Video - as fast as user's watching speed

If this is an offline system that doesn't interface individual humans and there are just batches of requests processed these metrics make no difference, but latency and throughput are the key ones.




### Simplified performance metrics

As you can tell the discussed above metrics have a lot of overlap in them. Practically we can reduce all of them to just these 2 metrics: Prefill throughput and Decode throughput - and probably how many parallel requests per second the system can handle.

#### Prefill throughput

This is how fast the system can pre-process the prompt - in tokens per second.

Assuming there is a negligible overhead of receiving and sending the request, in the absence of a queue where the incoming request gets immediately workedon [TTFT](#time-to-first-token) is really the number of tokens in the prompt divided by the prefill tokens per seconds plus the time to generate the first token (which we can ignore as it'd be very fast).

If there is a queue then prefill throughput isn't enough, because then TTFT can be much longer as one has to add the time the request spent in the queue.

#### Decode throughput

This is how fast the system generates response tokens - in tokens per second.

This addresses, both the throughput and Time Per Output Token metrics.

The response latency then is the number of tokens in the prompt divided by the prefill throughput plus the number of generated tokens divided by the decode throughput.


### More metric notes

#### Percentiles

If you read benchmarks and run into things like p50, p75, p90, p95 and p99 percentiles - these are statistical filters that give you the results based on the percentage of results that fit under (or over) a certain threshold. Even the same request is likely to take a slightly different response time when it gets re-run multiple times. So, for example, if 95% of the time a throughput was higher than a certain value - that would be a p95 percentile. That also would mean that 5% of the time the throughput was lower than that same threshold value. The higher the number next to `p`, the more difficult it is to achieve.

For example, let's look at partial output of a system loading report generated by [k6](https://github.com/grafana/k6) on an inference server:

```
http_req_duration..: avg=13.74s   min=12.54s  med=13.81s   max=13.83s   p(90)=13.79s   p(95)=13.83s
http_req_receiving.: avg=27.98µs  min=15.16µs med=21.6µs   max=98.13µs  p(90)=44.98µs  p(95)=59.2µs
http_req_sending...: avg=133.8µs  min=20.47µs med=75.39µs  max=598.04µs p(90)=327.73µs p(95)=449.65µs
```

If we look at the first line which reported the total generation time, if we look at the minimal recorded value of 12.54 seconds, we then know that 90% of responses took between 12.54 and 13.79 secs and 95% of responses took between
12.54 and 13.83 secs - and in this particular case the median reported value is between the p90 and p95 values.

The same interpretation applies to the other lines in the report, but the key exemplification here is that p90 values are lower than p95 values because time is being measured (the lower the better).

Percentiles are useful when outliers aren't important, so, for example, instead of looking at the slowest throughput measured you'd say ignore the worst 5% of outcomes and suddenly the system's performance looks much much better. But one has to be very careful with such discarding of bad outcomes when dealing with users, since it means that some of them will have a bad experience using your system. Also 5% translates to a whole lot of users if you have millions of them.

Please refer to [Percentile](https://en.wikipedia.org/wiki/Percentile) for a much more indepth explanation.




## Benchmarks

You can write your own benchmark as explained in [key inference performance metrics](#key-inference-performance-metrics) or use an existing one.

At the moment I use mainly the [prefill throughput](#prefill-throughput) and [decode throughput](#decode-throughput) benchmarks. The first one just measures tokens per second from the moment the request was sent and the first generated token received, and the second one is the throughput between the first and the last generated tokens received. Here is the relevant snippet of such measurement using [`openai` client completions API](https://github.com/openai/openai-python):

```
[... create client, data, etc. ...]
prefill_tokens_len = len(prompt)
start_time = time.time()
decode_text = ""
decode_started = False
completion = client.completions.create(prompt=prompt, ...)
for chunk in completion:
    if chunk.choices:
        decode_text += text
        if not decode_started:
            decode_started_time = time.time()
            prefill_time = decode_started_time - start_time
            decode_started = True

end_time = time.time()
decode_time = end_time - decode_started_time
decode_tokens = tokenizer.encode(decode_text)
decode_tokens_len = len(decode_tokens)

# tokens/per sec
prefill_throughput = prefill_tokens_len / prefill_time
decode_throughput  = decode_tokens_len  / decode_time
```

The `prefill_throughput` is not very precise here, since the client only know when it sent the request and received the first token, so a bit more went into this stage than pure prompt-preprocessing, but it should be close enough.

Of course, like any serious benchmark, you want to run this multiple times to get realistic numbers, as the variance  between single runs can be quite large.

note: I've discovered that when I use the openAI client it doesn't scale well and with many concurrent clients the client creates a bottleneck and doesn't measure the real server performance - I am yet to figure out if it's an issue in my code or the openAI client or how it interacts with vllm server - I'm investigating here https://github.com/vllm-project/vllm/issues/7935

Here are some good starting points for load testing:

- https://github.com/grafana/k6 - useful for load testing to simulate multiple concurrent clients - uses JavaScript clients.
- https://github.com/bentoml/llm-bench - benchmarks inference loads (not yet sure if it works only for BentoML)




## Inference frameworks

There are many dozens of inference frameworks and more emerging every week, so it'd be very difficult to list them all. So this here you will find a starter list of a handful of inference frameworks that might be a good fit for your needs, but do check out other frameworks if the ones listed here don't satisfy your needs.

This section is trying hard to be neutral and not recommend any particular frameworks, since even if I was able to try them all out, there is no way I could possible guess which framework will work best for which user/company.


### vLLM

[vLLM](https://github.com/vllm-project/vllm)

### DeepSpeed-FastGen

[DeepSpeed-FastGen](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen) from [the DeepSpeed team](https://github.com/microsoft/DeepSpeed).

### TensorRT-LLM

[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) (also integrated what used to be `FasterTransformer`)

Supports only NVIDIA gpus.

### TGI

[TGI](https://github.com/huggingface/text-generation-inference)

### SGLang

[SGLang](https://github.com/sgl-project/sglang)

### OpenPPL

[OpenPPL](https://github.com/OpenPPL/ppl.nn)

### LightLLM

[LightLLM](https://github.com/ModelTC/lightllm)

### LMDeploy

[LMDeploy](https://github.com/InternLM/lmdeploy)

### MLC-LLM

[MLC-LLM](https://github.com/mlc-ai/mlc-llm)

If your favourite inference framework isn't listed please make a PR and add it.



### Accelerator-specific frameworks

Most inference framework obviously support NVIDIA CUDA. Some support AMD ROCm and Intel Gaudi.

But there are accelerator-specific frameworks:

### Intel Gaudi, MAX, etc.

-  https://github.com/intel/intel-extension-for-transformers



### How to choose an inference framework

To choose the most suitable inference framework you need to answer at least the following questions:

1. Does the framework have the features that you need? Be careful here, some frameworks list that they support feature A, but when you try to use it it's not well integrated or works really slowly.
2. Does the framework have a permissive license that meets your current and future needs? In practice we have seen that frameworks with licenses that go against commercial use are likely to be rejected by the community. For example HF's TGI tried to charge for commercial use and it backfired - so its license got reverted to the original Apache 2.0 license and now they are trying to recover from being shunned by the community.
3. Does the framework have a thriving community of contributors? Go to the framework's github repo and check how many contributors it has - if it's very few I'd be concerned as thriving frameworks usually tend to invite contributions and that means that even if the core contributors don't have the time some feature, some contributors might do it for you.
4. Does the framework have a high adoption? github stars are often a good indication, but sometimes it can be hyped up via smart marketing moves. So seek out other signals - e.g. `Used by` count on the framework's repo's main page on github - these are real numbers. Lots of PRs and Issues is another flag. Then search the web for how many articles are written about the given framework.
5. Are the framework maintainers responsive to Issues and PRs? Some frameworks will ignore many Issues and even PRs. Check the count of how many PRs and Issues not being addressed. A high outstanding open Issues is a difficult signal - from one side it means this is a popular project, from the other side it means the developer team and contributors can't cope with the needs of its users.
6. While the majority of ML inference frameworks are written in Python, with some sprinkling of C++ or Triton for fused kernels, some aren't written in Python. (e.g. NVIDIA's TensorRT-LLM is 99% C++, TGI's big chunk is written in Rust). If something doesn't work the way you need it to and you filed an Issue and it's not being addressed, will you be able to get your hands dirty and modify the framework to do what you need?
7. The other issue you may run into is that some frameworks don't want your PRs where you implemented missing features or made improvements and then you will end up maintaining a fork, which can be extremely difficult if you want to continue syncing with the upstream and cause a lot of pain to your developers.
8. Run some sort of load [benchmarks](#benchmarks) for the desired workloads to know if the performance is adequate.
9. Will you want to choose the [best cost-effective accelerator](../compute/accelerator#high-end-accelerators-for-llmvlm-workloads) down the road or are you OK being locked in into a specific vendor? For example, a framework from NVIDIA isn't likely to support any other accelerators besides NVIDIA's. Same goes for AMD and Intel.

For example, here is a snapshot of [vllm](https://github.com/vllm-project/vllm)'s stats as of 2024-08-24, which is one of the most popular inference frameworks as of this writing.

![vllm](images/github-vllm-stats-2024-08-24.png)

You can see that it is used by many github repositories, it has a lot of contributors and that it's written mainly in Python. So it should be very easy to find this information about any inference framework you may consider. This was just an example and not an endorsement of vllm.


## Inference Chips

Besides general purpose accelerators some vendors have been working special ASICs that are designed to do Inference-only

### Groq

- [Groq](https://groq.com/)




## Resources

- [A Survey on Efficient Inference for Large Language Models (2024)](https://arxiv.org/abs/2404.14294)
