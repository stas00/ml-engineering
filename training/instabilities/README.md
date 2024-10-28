# Avoiding, Recovering From and Understanding Instabilities

Sub-sections:

* [Understanding Training Loss Patterns](training-loss-patterns.md) - types of spikes, divergences, grokking moments, resumes, etc.

## Learning from Training Logbooks

The best learning is to read [Publicly available training LLM/VLM logbooks](../../resources#publicly-available-training-llmvlm-logbooks) because there you can see exactly what happened and how the problem has been overcome.


## STD Init

Correctly initializing the initial distribution of the tensors can have a tremendous impact on training's stability. The `std` value isn't fixed and depends on the hidden dimension size.

This proved to be a very crucial setting in our pre-BLOOM 104B experiments and we couldn't break past the first few thousands iterations until we figured out that the 0.02 default `--init-method-std` in Megatron-LM was a way too big for our model.

We referred to these two sources:

1. "Transformers without Tears" paper https://arxiv.org/abs/1910.05895 prescribes: `sqrt(2/(NHIDDEN*5))`

2. The 530B training paper https://arxiv.org/abs/2201.11990 they used an even smaller init formula: `sqrt(1/(NHIDDEN*3))`

and decided to go with the 530B one as it leads to an even smaller init value.

To make it easier to compare the two formulas, they can be rewritten as:
1. `sqrt(0.4000/NHIDDEN)`
2. `sqrt(0.3333/NHIDDEN)`

Thus for `NHIDDEN=14336` the math was `sqrt(1/(14336*3)) = 0.00482` and that's what we used. It surely wasn't the only reason why we had no stability issues during BLOOM-176B training, but I think it was one of the crucial ones.


## Numerical instabilities

Certain mathematical operations could be unstable when dealing with low precision numbers.

For example, please see this very interesting [PyTorch guide on numerical stability](https://pytorch.org/docs/stable/notes/numerical_accuracy.html).

Now let's look at a specific example of this concept in action.

During 104B training experiments where fp16 mixed precision was used - the following improvement was proposed by [Corby Rosset](https://github.com/corbyrosset) to make [self-attention more stable](https://github.com/bigscience-workshop/Megatron-DeepSpeed/pull/118).

Specifically this [line](https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/c839a8aa30731f71b3738d56009be9668508e366/megatron/model/transformer.py#L303) shows that the `norm_factor` may be multiplied after the Query * Key matrix multiplication. If the dim of Q and K are very large, the output may blow up and the `norm_factor` won't be able to save it.

Proposal: move the `norm_factor` inward, so Q and K are scaled down before matrix multiply:
```
        matmul_result = torch.baddbmm(
            matmul_result,
            1.0/math.sqrt(self.norm_factor) * query_layer.transpose(0, 1),   # [b * np, sq, hn]
            1.0/math.sqrt(self.norm_factor) * key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0 if alibi is None else 1.0, alpha=1.0)

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)
```

To make the operation mathematically equivalent, moving the norm factor inward requires taking sqrt again
if n is a scalar, A and B matrices:
```
n * (A dot B) === (sqrt(n) * A) dot (sqrt(n) * B)
```

Now A and B dimensions can be significantly larger.

For CUDA kernel writers [CuBlas](https://docs.nvidia.com/cuda/cublas/index.html)'s `GemmStridedBatchedEx` at the time of this writing has a similar issue. It is defined as:

```
C+i*strideC=αop(A+i*strideA)op(B+i*strideB)+β(C+i*strideC), for i ∈[0,batchCount−1]
```

The issue is that `alpha` is multiplied after the matrix-matrix multiplication is done so it can cause instability.

## "Bad" combination of data batch and model parameter state

PaLM team observed dozens of loss spikes at "highly irregular intervals" when training larger models. While they were not able to track down the root cause, they mitigated the issue by restarting from an earlier checkpoint and skipping potentially problematic data batches. [Section 5.1 Training instability](https://arxiv.org/pdf/2204.02311.pdf)


## Time-domain correlation divergence in Adam

[A Theory on Adam Instability in Large-Scale Machine Learning](https://arxiv.org/abs/2304.09871) performs a rigorous study of divergence spikes while training LLMs at up to 546B parameters - and suggests that the time-domain correlation leads to divergence of Adam. This is triggered by the epsilon value not being small enough and gradient
estimation components become similar to the epsilon.

In section 7.1 they propose practical suggestions, the most interesting one of them is setting epsilon to 0 and possibly dealing with division by zero condition.
