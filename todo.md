# TODO

Also see [stabs](./stabs)

- re-run all-reduce bench and update plot+table as the bench switched to KiB/MiB/etc.
https://github.com/stas00/ml-engineering/tree/master/network/benchmarks#all_reduce-benchmark

- confirm which NCCL algorithm a multi-node `all-reduce` actually selects, because [suggestion 39](build/consistency-review-2026-07-27.md) is blocked on it. **This needs at least 2 nodes** - a single node has no inter-node traffic at all, so no amount of GPUs on one box can answer it. Two nodes is enough to identify the algorithm; reproducing the 73.9% table in [Inter-node speed depends on intra-node speed](https://github.com/stas00/ml-engineering/blob/master/network/README.md#inter-node-speed-depends-on-intra-node-speed) needs the original 4, since `(k-1)/(n-1)` is `3/31` at 4 nodes and `1/15` at 2. Using the multi-node recipe from [network/benchmarks/README.md](network/benchmarks/README.md), with debug logging added:

```bash
GPUS_PER_NODE=8
NNODES=2   # 4 to also reproduce the busbw table
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,TUNING \
python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:6000 \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --role `hostname -s`: \
    --tee 3 \
    all_reduce_bench.py 2>&1 | tee nccl-algo.txt
```

then `grep -iE "algo|proto|nvls|tree|ring|collnet" nccl-algo.txt | sort -u`. If it reports `NVLS`/`NVLSTree`/`Tree` at large payloads, the hierarchical model is confirmed and the `Multiple node training` worked example needs revising; if it reports `Ring` across all ranks, then 381.80GBps `busbw` at 16GiB needs a different explanation, since a flat ring would put 7.75GiB across a single 50GBps NIC per node hop.

  caveat on the interpretation, which matters as much as the run: NCCL's [env var docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) say only that "NVLS and NVLSTree enable NVLink SHARP offload", and never state how the two differ in scope. So reading `NVLSTree` as "NVLS within the node, tree between nodes" - which is what the hierarchical model needs - is an inference from the name, not documented behaviour. Settle that from the NCCL source (`src/graph/tuning.cc` for the selection logic, `src/device/` for the implementations) rather than from the algorithm name, otherwise 39 gets closed on a sound measurement plus an unsourced assumption.

- measure PCIe's achievable-vs-spec bandwidth ratio - nobody seems to publish it. Run `all_reduce_bench.py` with `NCCL_P2P_DISABLE=1` on a node whose topology is known (`nvidia-smi topo -m`), which forces the collective over PCIe, and compare the `busbw` against the x16 spec figure in the [PCIe table](https://github.com/stas00/ml-engineering/blob/master/network/README.md#pcie). We know NVLink lands at ~80% of spec (300GBps spec -> 235GBps measured); the equivalent PCIe number is currently unknown, so the PCIe rows in the intra-node tables can only be labelled theoretical. Note that [disable-nvlink.md](https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/results/disable-nvlink.md) does *not* answer this - it measures gpt2 training wall-clock (101s vs 131s), not bandwidth. Two refinements found on 2026-08-02: on EFA `NCCL_P2P_DISABLE=1` alone does not force the NIC path, because NCCL falls back P2P -> SHM -> network so `NCCL_SHM_DISABLE=1` is needed as well, and even then libfabric's EFA provider serves intra-node traffic from the instance's shared memory unless `FI_EFA_ENABLE_SHM_TRANSFER=0`. Also confirm GPUDirect RDMA is actually active, since NCCL disables it when the accelerator-to-NIC distance exceeds its threshold and then stages through host RAM, and on a virtualized instance ACS cannot be turned off and redirects PCIe peer-to-peer traffic through the CPU root complex unless the adapter has ATS enabled - each of these changes what the measurement means.

## Parked networking items - 2026-08-02

Small things left open at the end of the 2026-08-02 session. The two items above plus [suggestion 39](build/consistency-review-2026-07-27.md) and [suggestion 11](build/update-suggestions-2026-07-27.md) are the substantive ones; these are the loose ends.

- run `ib_write_bw -c SRD` on EFA once. The same section says RDMA-write-over-SRD was contributed to `perftest` by AWS and that the EFA path is unconfirmed. One test on any two EFA instances resolves it, after which the "unconfirmed here" footnote can go.

- reconcile the `perftest` stab. [stabs/incoming.md](stabs/incoming.md) still lists `perftest`/`ib_write_bw` as unwritten material, but the new section now covers part of it. Either fold the rest in or trim the stab.

- open question, not ours: how many EFA devices the `aws-ofi-nccl` plugin assigns to a single rank. [aws-ofi-nccl#890](https://github.com/aws/aws-ofi-nccl/issues/890) asks exactly this and has no replies. It matters because it decides whether a one-rank-per-node run measures one NIC, several NICs, or the accelerator's PCIe link - which is why that approach was rejected for the chapter in favour of `ib_write_bw`.

- optional: give model 2 in the hierarchical-arithmetic list the same general-form treatment models 1 and 3 got. Its `2*(32-1)/32 * 4GiB` now reads through the shared `P`/`g`/`k`/`n` symbols defined just above it, so this is cosmetic.
