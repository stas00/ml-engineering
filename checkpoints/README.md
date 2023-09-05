# Checkpoints

- [torch-checkpoint-convert-to-bf16](./torch-checkpoint-convert-to-bf16) - converts an existing fp32 torch checkpoint to bf16. If [safetensors](https://github.com/huggingface/safetensors/) are found those are converted as well. Should be easily adaptable to other similar use cases.

- [torch-checkpoint-shrink.py](./torch-checkpoint-shrink.py) - this script fixes checkpoints which for some reason stored tensors with storage larger than their view at the moment of saving. It clones the current view and re-saves them with just the storage of the current view.
