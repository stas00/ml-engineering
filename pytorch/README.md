# pytorch tools

- [pt-checkpoint-shrink.py](./pt-checkpoint-shrink.py) - this script fixes checkpoints which for some reason stored tensors with storage larger than their view at the moment of saving. It clones the current view and re-saves them with just the storage of the current view.
