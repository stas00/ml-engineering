# TODO

Also see [stabs](./stabs)

- re-run all-reduce bench and update plot+table as the bench switched to KiB/MiB/etc.
https://github.com/stas00/ml-engineering/tree/master/network/benchmarks#all_reduce-benchmark

- measure PCIe's achievable-vs-spec bandwidth ratio - nobody seems to publish it. Run `all_reduce_bench.py` with `NCCL_P2P_DISABLE=1` on a node whose topology is known (`nvidia-smi topo -m`), which forces the collective over PCIe, and compare the `busbw` against the x16 spec figure in the [PCIe table](https://github.com/stas00/ml-engineering/blob/master/network/README.md#pcie). We know NVLink lands at ~80% of spec (300GBps spec -> 235GBps measured); the equivalent PCIe number is currently unknown, so the PCIe rows in the intra-node tables can only be labelled theoretical. Note that [disable-nvlink.md](https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/results/disable-nvlink.md) does *not* answer this - it measures gpt2 training wall-clock (101s vs 131s), not bandwidth.
