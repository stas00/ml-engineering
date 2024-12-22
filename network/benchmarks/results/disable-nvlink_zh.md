# disable-nvlink - 中文翻译

# 禁用NVLink基准测试

让我们比较一下使用小样本维基文本训练gpt2语言模型的结果。

结果如下：

| NVlink | 时间 |
| -----  | ----: |
| Y      | 101秒 |
| N      | 131秒 |

可以看出，启用NVLink时训练速度快约23%。在第二个基准测试中，我们使用`NCCL_P2P_DISABLE=1`来告知GPU不使用NVLink，而是使用PCIe。

我们将使用[Hugging Face Transformers示例](https://github.com/huggingface/transformers/blob/58e3d23e97078f361a533b9ec4a6a2de674ea52a/examples/pytorch/language-modeling/run_clm.py)。

以下是完整的基准代码和输出：

```bash
# 使用NVLink的DDP

rm -r /tmp/test-clm; CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node 2 examples/pytorch/language-modeling/run_clm.py --model_name_or_path gpt2 \
--dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train \
--output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

{'train_runtime': 101.9003, 'train_samples_per_second': 1.963, 'epoch': 0.69}

# 不使用NVLink的DDP

rm -r /tmp/test-clm; CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 python -m torch.distributed.launch \
--nproc_per_node 2 examples/pytorch/language-modeling/run_clm.py --model_name_or_path gpt2 \
--dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train \
--output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

{'train_runtime': 131.4367, 'train_samples_per_second': 1.522, 'epoch': 0.69}
```

硬件：2x TITAN RTX，每块24GB，并配有2个NVLink (`NV2` 在 `nvidia-smi topo -m` 中)
软件：`pytorch-1.8-to-be` + `cuda-11.0` / `transformers==4.3.0.dev0`