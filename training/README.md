# Training

**Subsections**:

- [Model parallelism](model-parallelism)

- [Performance](performance)

- [Fault Tolerance](fault-tolerance)

- [Reproducibility](reproducibility)

- [Instabilities](instabilities)

- [Checkpoints](checkpoints)

- [Training hyper-parameters and model initializations](hparams.md)

- [Tensor precision / Data types](dtype.md)

- [Emulate a multi-node setup using just a single node](emulate-multi-node.md) - instructions on how to emulate a multi-node setup using just a single node - we use the `deepspeed` launcher here.

- [Re-train HF hub models from scratch using finetuning examples](re-train-hub-models.md)

- [Datasets](datasets.md)

**Tools**:

- [printflock.py](tools/printflock.py) - a tiny library that makes your `print` calls non-interleaved in a multi-gpu environment.

- [multi-gpu-non-interleaved-print.py](tools/multi-gpu-non-interleaved-print.py) - a `flock`-based wrapper around `print` that prevents messages from getting interleaved when multiple processes print at the same time - which is the case with `torch.distributed` used with multiple-gpus.
