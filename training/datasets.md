# Dealing with datasets

## Preprocessing and caching datasets on the main process

HF Accelerate has a very neat container [`main_process_first`](https://huggingface.co/docs/accelerate/v0.4.0/accelerator.html#accelerate.Accelerator.main_process_first) which allows to write code like:

```
with accelerator.main_process_first():
    # load and pre-process datasets
    dataset = datasets.load_dataset(...)
    # optionally cache it and have the rest of the processes load the cache
```
instead of the less intuitive and requiring code repetition:
```
if rank == 0:
    dataset = datasets.load_dataset(...)
dist.barrier()
if not rank == 0:
    dataset = datasets.load_dataset(...)
```

You want to download and process data on the main process and not all processes, because they will be all repeating the same thing in parallel and more over are likely to write to the same location which will result in interleaved broken result. It's also much faster IO-wise to serialize such work.

Now there is `main_process_first` and `local_main_process_first` - the first one is for when your data resides on a shared filesystem and all compute nodes can see it. The second one is for when the data is local to each node.

If you aren't using HF Accelerate, I have recreated similar containers, except called them:

- `global_main_process_first` - for shared fs
- `local_main_process_first` - for local to node fs

You can find them [here](tools/main_process_first.py).

Now, what if you want to write a generic code that automatically works on shared and local filesystems. I added another helper that automatically discovers what type of filesystem we are dealing with and based on that call the right containers. I called it `main_process_by_path_first`, which is used like:

```
path = "/path/to/data"
with main_process_by_path_first(path):
    # load and pre-process datasets
    dataset = datasets.load_dataset(...)
    # optionally cache it and have the rest of the processes load the cache
```

You can find it [here](tools/main_process_first.py).

Of course, besides containers you will also want utils to check the type of main process, and so there are 3 of those corresponding to the containers:

- `is_main_process_by_path(path)`
- `is_local_main_process()`
- `is_global_main_process()`

They are all found in [here](tools/main_process_first.py).

You can see them in action by running:

```
python -u -m torch.distributed.run --nproc_per_node=2 --rdzv_endpoint localhost:6000  --rdzv_backend c10d tools/main_process_first.py
```
