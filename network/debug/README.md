# Network Debug

Often you don't need to be a network engineer to figure out networking issues. Some of the common issues can be resolved by reading the following notes.



## Glossary

- OOB: Out-of-Band (typically a slower ethernet NIC)
- Bonding: using multiple NICs together for faster speed or as a back up
- IB: InfiniBand (Originally by Mellanox, acquired by NVIDIA)
- NIC: Network Interface Card


## How to diagnose NCCL multi-gpu and multi-node connectivity issues

This section is definitely non-exhaustive and is meant to cover some of the most common setup issues that I have often encountered. For more complex problems please research the [NCCL repo Issues](https://github.com/NVIDIA/nccl/issues) or file a new Issue if you can't find one matching your situation. NCCL also includes a brief [troubleshooting section](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2183/user-guide/docs/troubleshooting.html) but usually one learns a lot more from reading [Issues](https://github.com/NVIDIA/nccl/issues).

For the network diagnostics work, instead of using a full application which may take a long time to launch and have unrelated issue, I recommend using this specially developed design test script:  [torch-distributed-gpu-test.py](../../debug/torch-distributed-gpu-test.py).

First, run the nccl-based program after setting:

```
export NCCL_DEBUG=INFO
```
which will print a lot of debug info about the NCCL setup and its network traffic.

For example if you're using the aforementioned debug script, for a single node with 8 GPUs, you might do:

```
NCCL_DEBUG=INFO python -m torch.distributed.run --nproc_per_node 8 --nnodes 1 torch-distributed-gpu-test.py
```

To launch it on multiple nodes, you'd have to either use some orchestration software like SLURM or Kubernetes, or manually launch it on each node (`pdsh` would be of a huge help) - see the instructions inside [torch-distributed-gpu-test.py](../../debug/torch-distributed-gpu-test.py) for details. But to understand how things work I recommend starting with just 1 node and then progressing to 2, and later to more nodes.

Now, inspect the output of the program and look for a line that starts with:
```
NCCL INFO NET/
```
and then inspect which protocol and which interfaces it is using.

For example, this output:
```
NCCL INFO NET/FastSocket : Using [0]ibs108:10.0.19.12<0> [1]ibs109:10.0.19.13<0> [2]ibs110:10.0.19.14<0> [3]ibs111:10.0.19.15<0> [4]ibs112:10.0.19.16<0> [5]ibs113:10.0.19.17<0> [6]ibs114:10.0.19.18<0> [7]ibs115:10.0.19.19<0>
```

tells us that [nccl-fastsocket](https://github.com/google/nccl-fastsocket) transport layer plugin is used and it discovered 8 `ibs*` network interfaces (NIC cards). If you're using Google Cloud this is correct, and your NCCL is likely setup correctly. But if you're using InfiniBand (IB) and you're getting the above output, you're likely to clock a very low internode speed, because this means that you activated the wrong plugin.

In the case of IB, what you want to see is `NET/IB` and its IB interfaces:
```
NCCL INFO NET/IB : Using [0]mlx5_0:1/IB [1]mlx5_1:1/IB [2]mlx5_2:1/IB [3]mlx5_3:1/IB [4]mlx5_4:1/IB [5]mlx5_5:1/IB [6]mlx5_6:1/IB [7]mlx5_7:1/IB [RO]; OOB eno1:101.262.0.9<0>
```

Here, you can see that IB is used with 8 `mlx5_*` interfaces for collective comms, and one OOB, which stands for Out-Of-Band, and is used for doing bootstrapping the connections and is usually using a slower Ethernet NIC (at times [several NICs bonded into one](https://wiki.linuxfoundation.org/networking/bonding) - in case you're wondering what does `bond` in the interface name stand for).

To know which TCP/IP interfaces your node has you run `ifconfig` on one of the nodes (typically all similar nodes will have the same interface names, but not always).

If your collective comms network is IB, instead of `ifconfig` you'd run `ibstat`. The last example of `NCCL INFO NET` would correspond to the following output:

```
$ ibstat | grep mlx5
CA 'mlx5_0'
CA 'mlx5_1'
CA 'mlx5_2'
CA 'mlx5_3'
CA 'mlx5_4'
CA 'mlx5_5'
CA 'mlx5_6'
CA 'mlx5_7'
```

Since besides the fast inter-node connectivity NICs, you're also likely to have a slow management Ethernet NIC (or even several of those), that is there to be able to configure the node, use a shared file system, access the Internet, it's almost certain that `ifconfig` will also include additional NICs. Also you are likely to have a docker network interface, `lo` loopback and some others. For example on my desktop I may get the following output:

```
$ ifconfig
docker0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 172.99.0.1  netmask 255.255.0.0  broadcast 172.99.255.255
        inet6 f330::42:fe33:f335:7c94  prefixlen 64  scopeid 0x20<link>
        ether 02:42:fe:15:1c:94  txqueuelen 0  (Ethernet)
        RX packets 219909  bytes 650966314 (650.9 MB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 262998  bytes 20750134 (20.7 MB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0

lo: flags=73<UP,LOOPBACK,RUNNING>  mtu 65536
        inet 127.0.0.1  netmask 255.0.0.0
        inet6 ::1  prefixlen 128  scopeid 0x10<host>
        loop  txqueuelen 1000  (Local Loopback)
        RX packets 1147283113  bytes 138463231270 (138.4 GB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 1147283113  bytes 138463231270 (138.4 GB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0

eth0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 10.0.0.23  netmask 255.255.255.0  broadcast 10.0.0.255
        inet6 2601:3108:1c71:600:4224:7e4b:13e4:7b54  prefixlen 64  scopeid 0x0<global>
        ether 04:41:1a:16:17:bd  txqueuelen 1000  (Ethernet)
        RX packets 304675330  bytes 388788486256 (388.7 GB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 74956770  bytes 28501279127 (28.5 GB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0
        device memory 0xa3b00000-a3bfffff
```

The reason I mention all these is that the critical part is to ensure that NCCL reports only the correct interfaces in its `Using` debug line. If any interfaces like `docker0` or `lo` or `eth0` end up being reported, e.g.:

```
NCCL INFO NET/Socket : Using [0]eth0:10.0.0.23<0>
```

it's most likely not what you want if you have faster network interfaces available. But, of course, in some situations the Ethernet NIC is all you have, in which case the above is just fine - it'll be just very slow.

Sometimes, if the wrong interface ends up being used, the application might just hang.

If you have all the correct interfaces, plus some incorrect interfaces NCCL might work but at the slower speed.

If it's a cloud environment, typically your cloud provider should give you instructions on how to set things up correctly. If they didn't then you need to at least ask them which network interfaces you need to use to setup NCCL.

While NCCL tries hard to auto-discover which interfaces it should use, if it fails to do so correctly you can then help it by telling it which interfaces to use or not to use:

- `NCCL_SOCKET_IFNAME` can be used to specify which `ifconfig` interfaces to include or exclude when not using Infiniband. Here are some examples:

```
export NCCL_SOCKET_IFNAME=eth:        Use all interfaces starting with eth, e.g. eth0, eth1, …
export NCCL_SOCKET_IFNAME==eth0:      Use only interface eth0
export NCCL_SOCKET_IFNAME==eth0,eth1: Use only interfaces eth0 and eth1
export NCCL_SOCKET_IFNAME=^docker:    Do not use any interface starting with docker
export NCCL_SOCKET_IFNAME=^=docker0:  Do not use interface docker0.
```
The full doc is [here](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-socket-ifname).

- When using IB RDMA (IB Verbs interfaces), instead of `NCCL_SOCKET_IFNAME` use `NCCL_IB_HCA` env var which selects the interfaces for the collective communications. Examples:

```
export NCCL_IB_HCA=mlx5 :               Use all ports of all cards starting with mlx5
export NCCL_IB_HCA==mlx5_0:1,mlx5_1:1 : Use ports 1 of cards mlx5_0 and mlx5_1.
export NCCL_IB_HCA=^=mlx5_1,mlx5_4 :    Do not use cards mlx5_1 and mlx5_4.
```
The full doc is [here](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-ib-hca).

For example, often with IB, there will be additional interfaces like `mlx5_bond_0` which you don't want to be included in the NCCL comms. For example, this report would indicate that the wrong `[8]mlx5_bond_0:1/RoCE` interface was included and this would almost certainly lead to a low bandwidth:
```
NCCL INFO NET/IB : Using [0]mlx5_0:1/IB [1]mlx5_1:1/IB [2]mlx5_2:1/IB [3]mlx5_3:1/IB [4]mlx5_4:1/IB [5]mlx5_5:1/IB [6]mlx5_6:1/IB [7]mlx5_7:1/I [8]mlx5_bond_0:1/RoCE [RO]; OOB ibp25s0:10.0.12.82<0>
```
There you'd exclude it with:
```
export NCCL_IB_HCA=^mlx5_bond_0:1
```
or alternatively you could list explicitly the interfaces you want, e.g.:
```
export NCCL_IB_HCA==mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
```

As mentioned earlier using `ibstat` on one of the nodes interconnected with IB will show you the available IB interfaces.

Since NCCL tries to automatically choose the best network interfaces, you only need to do the above if NCCL doesn't work or it's slow. In normal circumstances NCCL should work out of the box, without the user needing to do anything special.

Additionally, depending on which cloud is used, it's very likely that the provider may give you a slew of environment variables to set. If you set some of them incorrectly, NCCL might work slowly or not work at all.

Another typical problem users run into is when they try to reuse their NCCL setup they had working on cloud A on a cloud B. Often things don't translate and one has to carefully remove any previously set environment variables and set them correctly anew for the new cloud. This issue is likely to occur even if you're using the same cloud, but different types of instances, as some network setups are very specific for a given instance and won't work elsewhere.

Once you think you have set up the NCCL correctly, the next thing is to benchmark your connectivity and ensure that it matches the advertised speed (well, ~80% of it). Proceed to the [benchmark chapter](../benchmarks).


## NCCL with docker containers

* Give enough resources by adding to the docker `run` these additional args: `–shm-size=1g –ulimit memlock=-1` ([more details](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2183/user-guide/docs/troubleshooting.html#sharing-data))
* Privileged access: sometimes you need to add `--privileged` to  the docker `run` args.
* Having the docker image include the right packages, e.g. if using IB you'd want at least to install `libibverbs1 librdmacm1`



## How to check if P2P is supported

Sometimes you need to know if the GPUs on your compute node support P2P access (Peer2Peer). Disabling P2P will typically lead to a slow intra-node connectivity.

You can see that on this particular 8x NVIDIA H100 node the P2P is supported:

```
$ nvidia-smi topo -p2p r
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7
 GPU0   X       OK      OK      OK      OK      OK      OK      OK
 GPU1   OK      X       OK      OK      OK      OK      OK      OK
 GPU2   OK      OK      X       OK      OK      OK      OK      OK
 GPU3   OK      OK      OK      X       OK      OK      OK      OK
 GPU4   OK      OK      OK      OK      X       OK      OK      OK
 GPU5   OK      OK      OK      OK      OK      X       OK      OK
 GPU6   OK      OK      OK      OK      OK      OK      X       OK
 GPU7   OK      OK      OK      OK      OK      OK      OK      X

Legend:

  X    = Self
  OK   = Status Ok
  CNS  = Chipset not supported
  GNS  = GPU not supported
  TNS  = Topology not supported
  NS   = Not supported
  U    = Unknown
```

On the other hand with this particular 2x NVIDIA L4 the P2P is not supported:
```
$ nvidia-smi topo -p2p r
        GPU0    GPU1
 GPU0   X       CNS
 GPU1   CNS     X
```

As you can see from the Legend,`CNS` signifies that "Chipset is not supported".

If you're using a high-end datacenter GPUs this is very unlikely to happen. Though some low-end datacenter GPUs may not support P2P like the example of L4 above.

For consumer-level GPUs there could be a variety of reasons for your GPU not being supported, often it's the IOMMU and/or ACS features being enabled. At other times it's just the driver version. And if you spend some time searching you might find someone hacking drivers to enable P2P in GPUs that shouldn't support P2P, like this [4090 P2P support repo](https://github.com/tinygrad/open-gpu-kernel-modules).

To check if PCI Access Control Services (ACS) are enabled and to disable those follow [this guide](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2183/user-guide/docs/troubleshooting.html#pci-access-control-services-acs).

IOMMU can be disabled in the BIOS.

You can also check P2P support between specific GPUs using torch - here are we checking for GPUs 0 and 1:

```
python -c "import torch; print(torch.cuda.can_device_access_peer(torch.device('cuda:0'), torch.device('cuda:1')))"
```
If there is no P2P support, the above would print `False`.



## How to count NCCL calls

Enable NCCL debug logging for subsystems - collectives:
```
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL
```

if you're working in a slurm environment with many nodes you probably want to perform this only on rank 0, like so:
```
if [[ $SLURM_PROCID == "0" ]]; then
  export NCCL_DEBUG=INFO
  export NCCL_DEBUG_SUBSYS=COLL
fi
```

Assuming your logs were all sent to `main_log.txt`, you can then count how many of each collective call were performed with:
```
grep -a "NCCL INFO Broadcast" main_log.txt     | wc -l
2590
grep -a "NCCL INFO AllReduce" main_log.txt     | wc -l
5207
grep -a "NCCL INFO AllGather" main_log.txt     | wc -l
1849749
grep -a "NCCL INFO ReduceScatter" main_log.txt | wc -l
82850
```

It might be a good idea to first isolate a specific stage of the training, as loading and saving will have a very different pattern from training iterations.

So I typically first slice out one iteration. e.g. if each iteration log starts with: `iteration: ...` then I'd first do:
```
csplit main_log.txt '/iteration: /' "{*}"
```
and then analyse one of the resulting files that correspond to the iterations. By default it will be named something like `xx02`.


## Useful NCCL Debug Environment Variables

The following env vars are most useful during debugging NCCL-related issues such as hanging and crashing. The full list of those can be found [here](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html).


### `NCCL_DEBUG`

This is the most commonly used env var to debug networking issues.

Values:
- `VERSION` - Prints the NCCL version at the start of the program.
- `WARN` - Prints an explicit error message whenever any NCCL call errors out.
- `INFO` - Prints debug information
- `TRACE` - Prints replayable trace information on every call.

For example:

```bash
NCCL_DEBUG=INFO python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 torch-distributed-gpu-test.py
```

This will dump a lot of NCCL-related debug information, which you can then search online if you find that some problems are reported.

And `NCCL_DEBUG_FILE` should be very useful when using `NCCL_DEBUG` as the information is copious especially if using many nodes.



### `NCCL_DEBUG_FILE`

When using `NCCL_DEBUG` env var, redirect all NCCL debug logging output to a file.

The default is `stdout`. When using many GPUs it can be very useful to save each process' debug info into its own log file, which can be done like so:

```
NCCL_DEBUG_FILE=/path/to/nccl-log.%h.%p.txt
```

- `%h` is replaced with the hostname
- `%p` is replaced with the process PID.

If you then need to analyse hundreds of these at once, here are some useful shortcuts:

- grep for a specific match and also print the file and line number where it was found:

```
grep -n "Init COMPLETE" nccl-log*
```

- show `tail -1` of all nccl log files followed by the name of each file

```
find . -name "nccl*" -exec sh -c 'echo "$(tail -1 "$1") ($1)"' _ {} \;
```



### `NCCL_DEBUG_SUBSYS`

`NCCL_DEBUG_SUBSYS` used in combination with `NCCL_DEBUG` tells the latter which subsystems to show. Normally you don't have to specify this variable, but sometimes the developers helping you may ask to limit the output to only some sub-systems, for example:

```
NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV,TUNING
```



### `NCCL_P2P_DISABLE`

Disables P2P comms - e.g. NVLink won't be used if there is one and the performance will be much slower as a result of that. Normally you don't want this but in a pinch sometimes this can be useful during debug.


### `NCCL_SOCKET_IFNAME`

This one is very useful if you have multiple network interfaces and you want to choose a specific one to be used.

By default NCCL will try to use the fastest type of an interface, which is typically `ib` (InfiniBand).

But say you want to use an Ethernet interface instead then you can override with:

```
NCCL_SOCKET_IFNAME=eth
```

This env var can be used at times to debug connectivity issues, if say one of the interfaces is firewalled, and perhaps the others aren't and can be tried instead. Or if you are not sure whether some problem is related to the network interface or something else, so it helps to test other interfaces to invalidate that the issue comes from network.
