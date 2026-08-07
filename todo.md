# TODO

Also see [stabs](./stabs)

Grouped by the hardware a task needs, since that is usually what blocks it. The `Parked networking items - 2026-08-02` group was dissolved into these sections on 2026-08-04.

## No hardware needed

- optional: give model 2 in the hierarchical-arithmetic list the same general-form treatment models 1 and 3 got. Its `2*(32-1)/32 * 4GiB` now reads through the shared `P`/`g`/`k`/`n` symbols defined just above it, so this is cosmetic.

## 1 node, 8x accelerators

- refresh the illustrative `busbw` table in [network/benchmarks/README.md](network/benchmarks/README.md) under `### all_reduce benchmark`, whose top figure is 91.69GBps from an earlier cluster. Left alone on 2026-08-04 because it does illustrate the output format correctly and the 4-node plot beside it matches - so this is a "is a current example better than an old one" call, not a correctness fix. A current 8x H200 plot and its environment now sit directly beneath it if you want to swap the table too.

- reference notes for any future attempt to force a collective onto the NIC path, which is harder than it looks: `NCCL_P2P_DISABLE=1` alone does not do it, because NCCL falls back P2P -> SHM -> network, so `NCCL_SHM_DISABLE=1` is needed as well, and even then libfabric's EFA provider serves intra-node traffic from the instance's shared memory unless `FI_EFA_ENABLE_SHM_TRANSFER=0`. Also confirm GPUDirect RDMA is actually active, since NCCL disables it when the accelerator-to-NIC distance exceeds its threshold and then stages through host RAM, and on a virtualized instance ACS cannot be turned off and redirects PCIe peer-to-peer traffic through the CPU root complex unless the adapter has ATS enabled - each of these changes what the measurement means.

## 2 nodes

All four items here were done on 2026-08-07 on a 4-node 8x H200 `p5en.48xlarge` allocation, and the section is kept only to record what was answered:

- **which algorithm a multi-node `all-reduce` selects** - `Ring` at 4 nodes, confirmed by forcing rather than by reading a log enum: `NCCL_ALGO=allreduce:ring` gave 364.65GBps `busbw` against the default's 364.87, while `allreduce:nvlstree` was available but 15% slower at 310.07. This closed review item `1` and opened item `73`, because the flat-ring model the chapter rejects turns out to fit its own measurements best once its one-NIC-per-hop premise is corrected.
- **NVLSTree at two nodes** - it is selected there (forced 463.29 against default 463.55), so the code behaves as `tuning.cc` says. But the number is useless: 2-node `busbw` came out at 486.80GBps against a *single* node's 482.05, i.e. faster than pure NVLink, which is impossible for a real inter-node measurement. NCCL's own model special-cases it - `min(bwIntra, nNodes <= 2 ? bwInter : bwInter/2)`. **Never characterise a fabric on two nodes.**
- **`ib_write_bw -c SRD` on EFA** - 193.72Gbps on one adapter, 96.9% of its 200Gbps line rate. The "unconfirmed here" footnote is gone. `perftest` needed `sudo apt-get install -y perftest` on both hosts, and without `-c SRD` the run dies at `Unable to create QP` since EFA has no RC transport.
- **aws-ofi-nccl#890** - partly answered. The node exposes 16 EFA devices at 200Gbps each, 2 per accelerator, 3200Gbps/400GBps per node - which confirms the chapter's `EFA v3 ... 16 200GbE` line. The plugin's *per-rank* device assignment was not captured before the allocation was released, so the upstream question is still open; a `NET/OFI` grep of an `NCCL_DEBUG=INFO` multi-node log would finish it.

## 4 nodes

- the `busbw` table in [Inter-node speed depends on intra-node speed](network/README.md#inter-node-speed-depends-on-intra-node-speed) was reproduced on H200 rather than the published B200: 1 node 482.05GBps against 4 nodes 369.06GBps at 16GiB, so leaving the node costs 1.31x where B200 costs 2.2x. That difference is the section's own thesis - both platforms have the same 400GBps per node, but H200's NVLink 4 is 450GBps against B200's NVLink 5 at 900GBps, so the closer the two fabrics are the less the node boundary costs. Worth adding as a second table, but held until item `73` settles what the section concludes.

## Specific hardware not currently to hand

- verify which collective algorithm the published B200 `busbw` rows actually ran, on a 4-node P6-B200 allocation. [Item `73`](build/consistency-review-2026-07-27.md) left the section honest but undecided: models 2 and 3 both fit the 22.05ms measurement within ~10%, and only the algorithm distinguishes them. One 4GiB run with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING` via `.deepspeed_env`, then `NCCL_ALGO=allreduce:ring` and `allreduce:nvlstree` compared against the default - about three minutes of node time. H200 measured `Ring`, but the AWS tuner keys off the instance type - its log says `base Tuner is chosen for platform: p5en.48xlarge` - so a P6-B200 allocation gets a different tuner table and the H200 result does not transfer.
- validate the SHARP/multicast granularity on an NVL36 or NVL72 system. [The SHARP section](network/README.md#sharp) now says the granularity there is *likely* 4 GPUs rather than 8, on the grounds that a compute tray is 2 GB200 modules = 4 GPUs and that the [partition guide](https://docs.nvidia.com/multi-node-nvlink-systems/partition-guide-v1-0.pdf) sizes partitions in fours and says partitions of `<=4` GPUs get no multicast benefit. What is actually measured is the 8x H200 HGX case: `NVLS` is not selected at 4 GPUs and the gain ramps 1.14x -> 1.29x from 5 to 8. Run the same `all_reduce_perf -g N` sweep with and without `NCCL_NVLS_ENABLE=0` on NVL hardware and the claim either firms up or gets corrected.

- [suggestion 3](build/update-suggestions-2026-07-27.md): add the P6e-GB200 row, blocked on reading its per-NIC rate off a live instance.
