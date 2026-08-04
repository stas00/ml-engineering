# TODO

Also see [stabs](./stabs)

- re-run all-reduce bench and update plot+table as the bench switched to KiB/MiB/etc.
https://github.com/stas00/ml-engineering/tree/master/network/benchmarks#all_reduce-benchmark

- confirm which NCCL algorithm a multi-node `all-reduce` actually selects, because [suggestion 39](build/consistency-review-2026-07-27.md) is blocked on it. Re-run the 4x B200 benchmark with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING` and grep for `algo|proto|nvls|tree|ring|collnet`. If it reports `NVLS`/`NVLSTree`/`Tree` at large payloads, the hierarchical model in [Inter-node speed depends on intra-node speed](https://github.com/stas00/ml-engineering/blob/master/network/README.md#inter-node-speed-depends-on-intra-node-speed) is confirmed and the `Multiple node training` worked example needs revising; if it reports `Ring` across all 32 ranks, then 381.80GBps `busbw` at 16GiB needs a different explanation, since a flat ring would put 7.75GiB across a single 50GBps NIC per node hop.

- measure PCIe's achievable-vs-spec bandwidth ratio - nobody seems to publish it. Run `all_reduce_bench.py` with `NCCL_P2P_DISABLE=1` on a node whose topology is known (`nvidia-smi topo -m`), which forces the collective over PCIe, and compare the `busbw` against the x16 spec figure in the [PCIe table](https://github.com/stas00/ml-engineering/blob/master/network/README.md#pcie). We know NVLink lands at ~80% of spec (300GBps spec -> 235GBps measured); the equivalent PCIe number is currently unknown, so the PCIe rows in the intra-node tables can only be labelled theoretical. Note that [disable-nvlink.md](https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/results/disable-nvlink.md) does *not* answer this - it measures gpt2 training wall-clock (101s vs 131s), not bandwidth. Two refinements found on 2026-08-02: on EFA `NCCL_P2P_DISABLE=1` alone does not force the NIC path, because NCCL falls back P2P -> SHM -> network so `NCCL_SHM_DISABLE=1` is needed as well, and even then libfabric's EFA provider serves intra-node traffic from the instance's shared memory unless `FI_EFA_ENABLE_SHM_TRANSFER=0`. Also confirm GPUDirect RDMA is actually active, since NCCL disables it when the accelerator-to-NIC distance exceeds its threshold and then stages through host RAM, and on a virtualized instance ACS cannot be turned off and redirects PCIe peer-to-peer traffic through the CPU root complex unless the adapter has ATS enabled - each of these changes what the measurement means.

## Parked networking items - 2026-08-02

Small things left open at the end of the 2026-08-02 session. The two items above plus [suggestion 39](build/consistency-review-2026-07-27.md) and [suggestion 11](build/update-suggestions-2026-07-27.md) are the substantive ones; these are the loose ends.

- verify `fi_pingpong`. The new [Measuring the inter-node fabric on its own](https://github.com/stas00/ml-engineering/blob/master/network/README.md#measuring-the-inter-node-fabric-on-its-own) section names it as the substitute for fabrics that use their own userspace stack rather than verbs, but no man page was ever opened and no flags are given. Either document a concrete invocation or drop the mention.

- run `ib_write_bw -c SRD` on EFA once. The same section says RDMA-write-over-SRD was contributed to `perftest` by AWS and that the EFA path is unconfirmed. One test on any two EFA instances resolves it, after which the "unconfirmed here" footnote can go.

- reconcile the `perftest` stab. [stabs/incoming.md](stabs/incoming.md) still lists `perftest`/`ib_write_bw` as unwritten material, but the new section now covers part of it. Either fold the rest in or trim the stab.

- open question, not ours: how many EFA devices the `aws-ofi-nccl` plugin assigns to a single rank. [aws-ofi-nccl#890](https://github.com/aws/aws-ofi-nccl/issues/890) asks exactly this and has no replies. It matters because it decides whether a one-rank-per-node run measures one NIC, several NICs, or the accelerator's PCIe link - which is why that approach was rejected for the chapter in favour of `ib_write_bw`.

- optional: give model 2 in the hierarchical-arithmetic list the same general-form treatment models 1 and 3 got. Its `2*(32-1)/32 * 4GiB` now reads through the shared `P`/`g`/`k`/`n` symbols defined just above it, so this is cosmetic.
