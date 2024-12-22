# re-train-hub-models - 中文翻译

# 从头重新训练HF Hub模型使用微调示例

HF Transformers 提供了非常棒的微调示例 https://github.com/huggingface/transformers/tree/main/examples/pytorch，涵盖了几乎所有的模态，并且这些示例开箱即用。

**但如果你希望从头开始重新训练而不是微调呢？**

这里有一个简单的技巧可以实现这个目标。

我们将使用 `facebook/opt-1.3b` 并计划使用 bf16 训练方案作为示例：

``` 
cat << EOT > prep-bf16.py
from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch

mname = "facebook/opt-1.3b"

config = AutoConfig.from_pretrained(mname)
model = AutoModel.from_config(config, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(mname)

path = "opt-1.3b-bf16"

model.save_pretrained(path)
tokenizer.save_pretrained(path)
EOT
```

现在运行：

``` 
python prep-bf16.py
```

这将会创建一个文件夹 `opt-1.3b-bf16`，其中包含了你需要从头开始训练模型的所有内容。换句话说，你有了一个类似预训练的模型，除了它只完成了初始化而没有进行任何训练。

根据上面的脚本调整以使用 `torch.float16` 或 `torch.float32` 如果你打算使用它们。

现在你可以像平常一样使用这个保存的模型进行微调：

``` 
python -m torch.distributed.run \
--nproc_per_node=1 --nnode=1 --node_rank=0 \
--master_addr=127.0.0.1 --master_port=9901 \
examples/pytorch/language-modeling/run_clm.py --bf16 \
--seed 42 --model_name_or_path opt-1.3b-bf16 \
--dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 \
--per_device_train_batch_size 12 --per_device_eval_batch_size 12 \
--gradient_accumulation_steps 1 --do_train --do_eval --logging_steps 10 \
--save_steps 1000 --eval_steps 100 --weight_decay 0.1 --num_train_epochs 1 \
--adam_beta1 0.9 --adam_beta2 0.95 --learning_rate 0.0002 --lr_scheduler_type \
linear --warmup_steps 500 --report_to tensorboard --output_dir save_dir
```

关键条目是：
``` 
--model_name_or_path opt-1.3b-bf16
```

其中 `opt-1.3b-bf16` 是你在上一步中刚刚生成的本地目录。

有时可以找到与原始模型相同的训练数据集，有时你必须使用替代数据集。

其余的超参数通常可以在与模型相关的论文或文档中找到。

总之，这个方法允许你使用微调示例来重新训练你可以在 [HF hub](https://huggingface.co/models) 上找到的任何模型。