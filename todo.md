# TODO

Also see [stabs](./stabs)

Grouped by the hardware a task needs, since that is usually what blocks it. The `Parked networking items - 2026-08-02` group was dissolved into these sections on 2026-08-04.

## No hardware needed

- reconcile the `perftest` stab. [stabs/incoming.md](stabs/incoming.md) still lists `perftest`/`ib_write_bw` as unwritten material, but the new section now covers part of it. Either fold the rest in or trim the stab. Partly addressed on 2026-08-04 - a pointer to [Measuring the inter-node fabric on its own](network/README.md#measuring-the-inter-node-fabric-on-its-own) was added above the list - so what remains is deciding whether `ib_write_bw` itself should come out of the list.

- optional: give model 2 in the hierarchical-arithmetic list the same general-form treatment models 1 and 3 got. Its `2*(32-1)/32 * 4GiB` now reads through the shared `P`/`g`/`k`/`n` symbols defined just above it, so this is cosmetic.

## 1 node, 8x accelerators

- re-run all-reduce bench and update plot+table as the bench switched to KiB/MiB/etc.
https://github.com/stas00/ml-engineering/tree/master/network/benchmarks#all_reduce-benchmark
  Note this was impossible between 2025-12-08 and 2026-08-04: `all_reduce_bench.py` passed `formatter_class` to `parse_args()` and so raised a `TypeError` on every invocation, on every Python version. Now fixed, and an 8x H200 run is in hand - the full 32KiB-16GiB `busbw`/`algbw` table plus a generated plot - so what is left is choosing what to publish and where.

- reference notes for any future attempt to force a collective onto the NIC path, which is harder than it looks: `NCCL_P2P_DISABLE=1` alone does not do it, because NCCL falls back P2P -> SHM -> network, so `NCCL_SHM_DISABLE=1` is needed as well, and even then libfabric's EFA provider serves intra-node traffic from the instance's shared memory unless `FI_EFA_ENABLE_SHM_TRANSFER=0`. Also confirm GPUDirect RDMA is actually active, since NCCL disables it when the accelerator-to-NIC distance exceeds its threshold and then stages through host RAM, and on a virtualized instance ACS cannot be turned off and redirects PCIe peer-to-peer traffic through the CPU root complex unless the adapter has ATS enabled - each of these changes what the measurement means.

## 2 nodes

- confirm which NCCL algorithm a multi-node `all-reduce` actually selects, because [suggestion 39](build/consistency-review-2026-07-27.md) is blocked on it. A single node has no inter-node traffic at all, so no amount of GPUs on one box can answer it. Two nodes is enough to identify the algorithm; reproducing the 73.9% table needs 4 - see the 4-node section. Using the multi-node recipe from [network/benchmarks/README.md](network/benchmarks/README.md), with debug logging added:

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

  the interpretation is now settled, which is what makes the run above conclusive rather than suggestive. `NCCL_ALGO_NVLS_TREE` in `src/graph/tuning.cc` (v2.27.7) is explicitly gated off on one node - `// Disable NVLS Tree on a single node` / `if (comm->nNodes == 1 && a == NCCL_ALGO_NVLS_TREE) disable = 1;` - its bandwidth model is bounded by both fabrics, `min(bwIntra, nNodes <= 2 ? bwInter : bwInter/2)`, and its latency model is `intraLat + 2 * log2i(nNodes) * interLat`: one intra-node step plus a log-depth tree across nodes. Compare plain `NVLS`, which is `intraLat` plus a single `interLat`. So `NVLSTree` really is NVLS inside each node with a tree between nodes, and a reported `NVLSTree` at large payloads does mean the hierarchical model rather than a flat ring.

- while running the sweep above, confirm `NVLSTree` does get selected once there are two nodes. The source says it must - `tuning.cc` disables it outright when `nNodes == 1`, which matches a single 8x H200 node on 2026-08-04 where it was offered in the tuning table and chosen zero times, `NVLS` winning every payload from 2MiB up and `RING` below. So this is a check that the model behaves as its code says, not an open question.

- run `ib_write_bw -c SRD` on EFA once. The [Measuring the inter-node fabric on its own](network/README.md#measuring-the-inter-node-fabric-on-its-own) section says RDMA-write-over-SRD was contributed to `perftest` by AWS and that the EFA path is unconfirmed. One test on any two EFA instances resolves it, after which the "unconfirmed here" footnote can go. The `stas-dev-2` H200 nodes qualify - they expose 16 `rdmap*` devices, i.e. EFA - but `perftest` is not installed there and has to be built from source.

- answer [aws-ofi-nccl#890](https://github.com/aws/aws-ofi-nccl/issues/890) ourselves rather than waiting: how many EFA devices the `aws-ofi-nccl` plugin assigns to a single rank. It has no upstream replies. A multi-node `NCCL_DEBUG=INFO` log reports the `NET/OFI` device assignment directly, so the 2-node run above should answer it as a side effect. It matters because it decides whether a one-rank-per-node run measures one NIC, several NICs, or the accelerator's PCIe link - which is why that approach was rejected for the chapter in favour of `ib_write_bw`.

## 4 nodes

- reproduce the `busbw` table in [Inter-node speed depends on intra-node speed](network/README.md#inter-node-speed-depends-on-intra-node-speed). The conversion is `(k-1)/(n-1)`, which is `3/31` = 9.7% at 4 nodes and `1/15` = 6.7% at 2, so the published 73.9% figures need the original 4 nodes. Two nodes identifies the algorithm but will not reproduce the numbers.

## Specific hardware not currently to hand

- validate the SHARP/multicast granularity on an NVL36 or NVL72 system. [The SHARP section](network/README.md#sharp) now says the granularity there is *likely* 4 GPUs rather than 8, on the grounds that a compute tray is 2 GB200 modules = 4 GPUs and that the [partition guide](https://docs.nvidia.com/multi-node-nvlink-systems/partition-guide-v1-0.pdf) sizes partitions in fours and says partitions of `<=4` GPUs get no multicast benefit. What is actually measured is the 8x H200 HGX case: `NVLS` is not selected at 4 GPUs and the gain ramps 1.14x -> 1.29x from 5 to 8. Run the same `all_reduce_perf -g N` sweep with and without `NCCL_NVLS_ENABLE=0` on NVL hardware and the claim either firms up or gets corrected.

- [suggestion 11](build/update-suggestions-2026-07-27.md): add the P6e-GB200 row, blocked on reading its per-NIC rate off a live instance.
