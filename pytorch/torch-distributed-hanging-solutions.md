# Solutions to torch.distributed Hanging

Try to use the following script [torch-distributed-gpu-test.py](torch-distributed-gpu-test.py) to diagnose the situation.

## Various things to try to resolve the hanging:

Run the script with `NCCL_DEBUG=INFO` env var and try to study the outcome for obvious errors. It will tell you which device it's using, e.g.:
```
DeepWhite:21288:21288 [0] NCCL INFO NET/Socket : Using [0]enp67s0:192.168.50.21<0>
```
So it's using interface `enp67s0` over `192.168.50.21`

Is your `192.168.50.21` firewalled? or is it somehow a misconfigured network device?

Does it work if you use a loopback device `127.0.0.1`?
```
NCCL_DEBUG=INFO NCCL_SOCKET_IFNAME=lo python -m torch.distributed.run --nproc_per_node 4 --nnodes 1 torch-distributed-gpu-test.py
```

if not, see what other local network devices you have via `ifconfig` - try that instead of `lo` if any.

It's currently using `enp67s0` in your case.

---

If not, does it work if you use the first 2 or the last 2 gpus

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 torch-distributed-gpu-test.py
```
then the 2nd pair:
```
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 torch-distributed-gpu-test.py
```
----

If not, attach to each process with `sudo py-spy dump -n -p PID` after `pip install py-spy`. `PID` is the process id of the hanging python process.

and see where it hangs

---

Some AMD users may need to [Disable IOMMU](https://github.com/stas00/toolbox/issues/1#issuecomment-1076830400)
