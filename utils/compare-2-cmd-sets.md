# How to sort and format cmd args to make an easier sense of them and to be able to compare them

## Perl to help

This little one liner will take any command line and will break it up into one arg per line and sort them.

```
$ perl -le '$_=join " ", @ARGV; s/$/ /; s/=/ /g; s/ +/ /g; ($s,@a)=split /(?=--)/, $_; print join "\\\n", ($s, sort @a)'
```

## Find a diff between two cmd sets

So let's say you want to find out what's different between:
```
python -m torch.distributed.launch --nproc_per_node=2 --master_port=9910  ./finetune_trainer.py \
--model_name_or_path sshleifer/distill-mbart-en-ro-12-4 --output_dir output_dir --adam_eps 1e-06 \
--data_dir wmt_en_ro --do_train --freeze_embeds --label_smoothing 0.1 --learning_rate 3e-5 \
--logging_first_step --logging_steps 1000 --max_source_length 128 --fp16 --max_target_length 128 \
--num_train_epochs 1 --overwrite_output_dir --per_device_train_batch_size $BS --sortish_sampler \
--src_lang en_XX --task translation --tgt_lang ro_RO --val_max_target_length 128 --warmup_steps 500 \
--n_train 500 --sharded_ddp 
```
and:
```
python -m torch.distributed.launch --nproc_per_node=2 --master_port=9910  ./finetune_trainer.py \
--model_name_or_path sshleifer/distill-mbart-en-ro-12-4 --output_dir output_dir --adam_eps 1e-06 \
--data_dir wmt_en_ro --do_train --freeze_embeds --label_smoothing 0.1 --learning_rate 3e-5 \
--logging_first_step --logging_steps 1000 --max_source_length 128 --max_target_length 128 \
--num_train_epochs 1 --overwrite_output_dir --per_device_train_batch_size $BS --sortish_sampler \
--src_lang en_XX --task translation --tgt_lang ro_RO --val_max_target_length 128 --warmup_steps 500 \
--n_train 500 --sharded_ddp
```

We run each cmd set through the one liner, passing the cmd as its argument and save the results in `bad` and `good`
```
$ perl -le '$_=join " ", @ARGV; s/$/ /; s/=/ /g; s/ +/ /g; ($s,@a)=split /(?=--)/, $_; print join "\\\n", ($s, sort @a)' "python -m torch.distributed.launch --nproc_per_node=2 --master_port=9910  ./finetune_trainer.py --model_name_or_path sshleifer/distill-mbart-en-ro-12-4 --output_dir output_dir --adam_eps 1e-06 --data_dir wmt_en_ro --do_train --freeze_embeds --label_smoothing 0.1 --learning_rate 3e-5 --logging_first_step --logging_steps 1000 --max_source_length 128 --fp16 --max_target_length 128 --num_train_epochs 1 --overwrite_output_dir --per_device_train_batch_size $BS --sortish_sampler --src_lang en_XX --task translation --tgt_lang ro_RO --val_max_target_length 128 --warmup_steps 500 --n_train 500 --sharded_ddp " > bad
$ perl -le '$_=join " ", @ARGV; s/$/ /; s/=/ /g; s/ +/ /g; ($s,@a)=split /(?=--)/, $_; print join "\\\n", ($s, sort @a)' "python -m torch.distributed.launch --nproc_per_node=2 --master_port=9910  ./finetune_trainer.py --model_name_or_path sshleifer/distill-mbart-en-ro-12-4 --output_dir output_dir --adam_eps 1e-06 --data_dir wmt_en_ro --do_train --freeze_embeds --label_smoothing 0.1 --learning_rate 3e-5 --logging_first_step --logging_steps 1000 --max_source_length 128 --max_target_length 128 --num_train_epochs 1 --overwrite_output_dir --per_device_train_batch_size $BS --sortish_sampler --src_lang en_XX --task translation --tgt_lang ro_RO --val_max_target_length 128 --warmup_steps 500 --n_train 500 --sharded_ddp" > good
```

Now we can diff
```
$ diff bad good
5d4
< --fp16 \
```

voila - the difference is `--fp16`

What's inside:
```
$ cat bad
python -m torch.distributed.launch \
--adam_eps 1e-06 \
--data_dir wmt_en_ro \
--do_train \
--fp16 \
--freeze_embeds \
--label_smoothing 0.1 \
--learning_rate 3e-5 \
--logging_first_step \
--logging_steps 1000 \
--master_port 9910 ./finetune_trainer.py \
--max_source_length 128 \
--max_target_length 128 \
--model_name_or_path sshleifer/distill-mbart-en-ro-12-4 \
--n_train 500 \
--nproc_per_node 2 \
--num_train_epochs 1 \
--output_dir output_dir \
--overwrite_output_dir \
--per_device_train_batch_size 12 \
--sharded_ddp \
--sortish_sampler \
--src_lang en_XX \
--task translation \
--tgt_lang ro_RO \
--val_max_target_length 128 \
--warmup_steps 500
```
Which was generated from:
```
```
python -m torch.distributed.launch --nproc_per_node=2 --master_port=9910  ./finetune_trainer.py \
--model_name_or_path sshleifer/distill-mbart-en-ro-12-4 --output_dir output_dir --adam_eps 1e-06 \
--data_dir wmt_en_ro --do_train --freeze_embeds --label_smoothing 0.1 --learning_rate 3e-5 \
--logging_first_step --logging_steps 1000 --max_source_length 128 --fp16 --max_target_length 128 \
--num_train_epochs 1 --overwrite_output_dir --per_device_train_batch_size $BS --sortish_sampler \
--src_lang en_XX --task translation --tgt_lang ro_RO --val_max_target_length 128 --warmup_steps 500 \
--n_train 500 --sharded_ddp 
```
```
