# README - 中文翻译

以下是一些完整的SLURM脚本，展示了如何将各种启动器与使用`torch.distributed`的软件集成（但应该可以轻松适应其他分布式环境）。

- [torchrun](torchrun-launcher.slurm) - 用于[PyTorch分布式](https://github.com/pytorch/pytorch)。
- [accelerate](accelerate-launcher.slurm) - 用于[HF Accelerate](https://github.com/huggingface/accelerate)。
- [lightning](lightning-launcher.slurm) - 用于[Lightning](https://lightning.ai/)（“PyTorch Lightning”和“Lightning Fabric”）。
- [srun](srun-launcher.slurm) - 用于原生SLURM启动器——在这里我们需要手动预设`torch.distributed`期望的环境变量。

所有这些脚本都使用[demo脚本torch-distributed-gpu-test.py](../../../debug/torch-distributed-gpu-test.py)，你可以通过以下命令将其复制到当前目录：
``` 
cp ../../../debug/torch-distributed-gpu-test.py .
```
假设你已经克隆了这个仓库。但是你可以将其替换为你需要的任何其他脚本。