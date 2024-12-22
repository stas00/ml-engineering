# README - 中文翻译

- [torch-checkpoint-convert-to-bf16](./torch-checkpoint-convert-to-bf16) - 将现有的fp32 PyTorch检查点转换为bf16。如果找到[safetensors](https://github.com/huggingface/safetensors/)，也会一并转换。应该可以轻松适应其他类似的用例。

- [torch-checkpoint-shrink.py](./torch-checkpoint-shrink.py) - 该脚本修复了由于某些原因在保存时存储的张量存储空间大于其视图的检查点。它克隆当前视图，并仅使用当前视图的存储重新保存它们。