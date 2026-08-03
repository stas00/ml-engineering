# TODO

Also see [stabs](./stabs)

- re-run all-reduce bench and update plot+table as the bench switched to KiB/MiB/etc.
https://github.com/stas00/ml-engineering/tree/master/network/benchmarks#all_reduce-benchmark

- confirm which NCCL algorithm a multi-node `all-reduce` actually selects, because [suggestion 39](build/consistency-review-2026-07-27.md) is blocked on it. Re-run the 4x B200 benchmark with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING` and grep for `algo|proto|nvls|tree|ring|collnet`. If it reports `NVLS`/`NVLSTree`/`Tree` at large payloads, the hierarchical model in [Inter-node speed depends on intra-node speed](https://github.com/stas00/ml-engineering/blob/master/network/README.md#inter-node-speed-depends-on-intra-node-speed) is confirmed and the `Multiple node training` worked example needs revising; if it reports `Ring` across all 32 ranks, then 381.80GBps `busbw` at 16GiB needs a different explanation, since a flat ring would put 7.75GiB across a single 50GBps NIC per node hop.

- measure PCIe's achievable-vs-spec bandwidth ratio - nobody seems to publish it. Run `all_reduce_bench.py` with `NCCL_P2P_DISABLE=1` on a node whose topology is known (`nvidia-smi topo -m`), which forces the collective over PCIe, and compare the `busbw` against the x16 spec figure in the [PCIe table](https://github.com/stas00/ml-engineering/blob/master/network/README.md#pcie). We know NVLink lands at ~80% of spec (300GBps spec -> 235GBps measured); the equivalent PCIe number is currently unknown, so the PCIe rows in the intra-node tables can only be labelled theoretical. Note that [disable-nvlink.md](https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/results/disable-nvlink.md) does *not* answer this - it measures gpt2 training wall-clock (101s vs 131s), not bandwidth.
