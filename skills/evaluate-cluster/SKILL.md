---
name: evaluate-cluster
description: >-
  Evaluates an ML GPU cluster for a cloud trial or acceptance test: environment
  dump, isolated newest PyTorch, matmul FLOPS (MAMF/MSMF) on every GPU while the
  others compute, intra-node all-reduce bandwidth and per-call latency of every collective,
  the same inter-node on every node you were given (omit those sections if there is only
  one node), fio on local disk and shared FS,
  dated markdown report. Use when the user asks to evaluate a cluster, kick the
  tires on trial nodes, run cluster acceptance, or measure GPU/network/storage.

  Canonical copy: https://github.com/stas00/ml-engineering/blob/master/skills/evaluate-cluster/SKILL.md
---

# Evaluate a GPU cluster

**Always use the GitHub copy if it differs from this file.** People pass around downloads that can be months or years old.

- This skill: https://github.com/stas00/ml-engineering/blob/master/skills/evaluate-cluster/SKILL.md
- Raw: https://raw.githubusercontent.com/stas00/ml-engineering/master/skills/evaluate-cluster/SKILL.md

Before doing any eval work, `curl` the raw URL. If it is not byte-identical to the file you are reading, **stop following this copy** and follow the GitHub file instead (fetch it, then continue from that text). If GitHub is unreachable, say so and continue with this file.

This file is the whole runbook. You do not need any other document from the author's books. Scripts and spec tables are on GitHub; fetch them (commands below). Optional background: [How to evaluate the cluster](https://github.com/stas00/ml-engineering/blob/master/insights/how-to-choose-cloud-provider.md#how-to-evaluate-the-cluster).

## Keep this skill up to date

**Every correction the user makes to a report is a correction to this skill.** When they rename a heading, reject a phrase, ask for a different TOC label, or point out an assumption, fix the report **and** edit this file in the same turn so the next eval starts from the corrected rule. A fix that lives only in one report is a fix you will be asked for again.

- Write the rule where the next agent will hit it — the section that produces that text, not a footnote.
- Encode the **general principle**, not just the one string. "Don't write `Skip reasons: none`" became "never assert an absence".
- When the correction changes a heading, update the heading rule, the TOC example, **and** every cross-reference to the old name in this file.
- Do not report the edit as done until you have grepped this file for the old wording.

Reports under `reports/` other than the one you are writing are historical. Leave them alone.

Refuse to invent access, GPU counts, filesystem paths, or peer IPs. **Preflight first**: list what you have, **ask for every missing item, and wait**. Do not start DCGM, MAMF, `fio-scan`, or `torch-dist-bench.py` until the checklist is complete and — when **≥2 nodes** — inter-node reachability has passed. Eval **all nodes the user gave** (1, 2, 4, 16, …). Inter-node all-reduce uses **every** one of those nodes (`NNODES × GPUs_per_node` ranks). **One node:** do not write an Inter-node section at all (no “Skipped”). If ≥2 nodes but inter-node still cannot run after the user has answered, omit the section and put the reason in **### Gaps** — never invent busbw.

## Terms

| term                 | meaning                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MAMF**             | Maximum Achievable Matmul FLOPS: the highest any **matmul shape** (M×N×K) reaches on that GPU. The winning shapes are short bursts — the clock sits at boost and the board stays far below its power limit — so this is a ceiling, not a rate any sustained workload holds.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| **MSMF**             | Maximum Sustainable Matmul FLOPS: what is left once the matmul shape is big enough to pin the board at its power limit and the clock settles below boost. This is what back-to-back dense matmuls actually get, provided the other GPUs of the node compute at the same time: they share the board's power and cooling budget, so measured next to idle GPUs it reads high. Both come from [`mamf-finder.py`](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/benchmarks/mamf-finder.py), which times on the order of 1500–2200 matmul shapes per GPU (the `auto` search adapts, so the count differs per run) and reports the W and MHz behind each winner. [`mamf-finder-all-gpus.py`](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/benchmarks/mamf-finder-all-gpus.py) runs it on every GPU of a node, one at a time, while all the other GPUs run a continuous matmul. |
| **DCGM**             | NVIDIA Data Center GPU Manager. `dcgmi diag -r 2` checks GPU memory, bandwidth, PCIe — not FLOPS.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| **lemon GEMM**       | A one-off inline `torch.matmul` (not a published script) on every GPU with the same large shape, to catch a dead, slow, or throttled card.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| **straggler**        | A GPU that works correctly but runs slower than its peers, so that every GPU of a synchronous job waits for it. Found by [`gpu-straggler-finder.py`](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/benchmarks/gpu-straggler-finder.py), not by DCGM or MAMF/MSMF. |
| **all-reduce**       | Collective: every rank ends up with the sum of all ranks' tensors. Used to stress NVLink and the network.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| **busbw / algbw**    | Printed by [`torch-dist-bench.py`](https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/torch-dist-bench.py). **busbw** = unidirectional bus bandwidth (what the fabric moves). **algbw** = algorithm bandwidth (what the caller sees). busbw is already unidirectional ([why](https://github.com/stas00/ml-engineering/blob/master/network/README.md#unidirectional-vs-bidirectional-duplex)).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| **NVLS**             | NVLink SHARP: the NVSwitch does the all-reduce in the switch. Typical gain vs ring busbw ~30% intra-node / ~25% inter-node. Does **not** speed up all-gather or reduce-scatter. Different from InfiniBand switch SHARP.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| **NCCL**             | NVIDIA Collective Communications Library (the GPU collective stack torch uses).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| **rdzv**             | torch.distributed **rendezvous**: a TCP store so ranks find `MASTER_ADDR:PORT` before NCCL starts.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| **hostfile**         | Text file, one peer address per line (`10.0.0.1` or `10.0.0.1 slots=8`). DeepSpeed `-H` and `pdsh -w` read it.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| **pdsh**             | Parallel SSH: run one command on many hosts.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| **fio-scan**         | Wrapper around `fio` ([script](https://github.com/stas00/ml-engineering/blob/master/storage/fio-scan)): six timed runs (16 KiB / 1 MiB / 1 GiB × read/write). Needs [`fio-json-extract.py`](https://github.com/stas00/ml-engineering/blob/master/storage/fio-json-extract.py) in the **same directory**.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| **`local/shared ×`** | local-disk bandwidth ÷ same-row shared-FS bandwidth — how many times slower the shared filesystem is. One column is enough: with a fixed block size the IOPS ratio is identical. Never a percentage, never `%local`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| **SM**               | GPU streaming multiprocessor. Count from `torch.cuda.get_device_properties(0).multi_processor_count`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| **HBM**              | GPU high-bandwidth memory. Report `nvidia-smi` `memory.total` in **GiB**.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| **TDP**              | Thermal design power: the GPU’s advertised watt limit (`power.limit`).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| **EFA**              | AWS Elastic Fabric Adapter (RDMA NICs named `rdmap*`). `ibstat` is often **empty**; use `rdma link`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| **overlay**          | Container union filesystem. `df -T /` may say `overlay`; the **backing** FS (xfs, ext4) is the partition type to report.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |

## Scripts (fetch, do not invent)

```bash
RAW=https://raw.githubusercontent.com/stas00/ml-engineering/master
curl -fsSL "$RAW/compute/accelerator/benchmarks/mamf-finder.py" -o mamf-finder.py
curl -fsSL "$RAW/compute/accelerator/benchmarks/mamf-finder-all-gpus.py" -o mamf-finder-all-gpus.py
# mamf-finder-all-gpus.py runs ./mamf-finder.py from its own directory — keep both in one directory.
curl -fsSL "$RAW/network/benchmarks/torch-dist-bench.py" -o torch-dist-bench.py
curl -fsSL "$RAW/compute/accelerator/benchmarks/gpu-straggler-finder.py" -o gpu-straggler-finder.py
curl -fsSL "$RAW/debug/torch-distributed-gpu-test.py" -o torch-distributed-gpu-test.py
curl -fsSL "$RAW/storage/fio-scan" -o fio-scan
curl -fsSL "$RAW/storage/fio-json-extract.py" -o fio-json-extract.py
chmod +x fio-scan
# fio-scan calls `python ./fio-json-extract.py` — keep both in one directory.
```

Official BF16 peaks: [TFLOPS comparison table](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/README.md#tflops-comparison-table). NVLink gens: [NVLink](https://github.com/stas00/ml-engineering/blob/master/network/README.md#nvlink). Intra-node all-to-all: [bandwidth](https://github.com/stas00/ml-engineering/blob/master/network/README.md#all-to-all-bandwidth). Inter-node adapters: [networking](https://github.com/stas00/ml-engineering/blob/master/network/README.md#inter-node-networking).

Copy artifacts **off the cluster immediately** into `reports/raw-<name>/` **next to this skill** (or the working directory you chose). Inline every plot: `![alt](raw-<name>/file.png)` — never link-only. Cluster access and `/tmp` disappear.

**Links in the report** to those scripts and tables are absolute GitHub URLs (`https://github.com/stas00/ml-engineering/blob/master/…`). Never `../` relatives — the report may be copied elsewhere. Sibling `raw-<name>/` stays relative. External vendor/kernel docs stay as-is.

**Every GitHub-hosted tool the run used** is introduced and linked in that section's `### Introduction` — one clause on what it does, then later mentions are bare `code`. The reader of the report must not meet a tool they cannot click. Do not write “the book” or “MLE” in the report; the GitHub URL is the source.

**Every section `### Introduction`:** numbered list (`1.` `2.` …) of passes that ran, `tool — what it does / where`. Same shape in A, B, and C.

Tools not in the fetch list: `dcgmi diag` is NVIDIA's; the lemon GEMM is an inline script — say so.

**Do not leak eval-process shorthand into the report.** The report reader was not in your session. Avoid “kept”, “as configured”, “we left it”, “platform value”, “the fix”, “rerun”. Name the actor and the state: *“the container image presets `NCCL_NVLS_ENABLE=1`, and every benchmark ran with it unchanged.”*

## Preflight (hard gate)

**Goal:** be able to run the rest of the eval without blocking on the user. Slow work (PyTorch wheels, DCGM `-r 2`, MAMF/MSMF, shared-FS `fio-scan`, GPU all-reduce) starts only after this gate.

### Ask now, not later

Collect and **write down** every row. Missing → message the user with the exact list, then **stop**.

| need                                       | why                            | typical answer                                                                                                               |
| ------------------------------------------ | ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| How to exec on a node                      | every command                  | `kubectl` (ask for context, namespace, pod/job), `ssh`, `srun`, or whatever CLI the user names                               |
| **All** node identities                    | lemon GEMM + inter-node        | every pod/hostname, not only rank 0                                                                                          |
| **Hostfile (or equivalent) with peer IPs** | inter-node SSH / rdzv / `pdsh` | path or pasted contents: one IP per line. Required when **≥2 nodes**. SLURM: `scontrol show hostnames` plus IPs.             |
| SSH / launcher details                     | `pdsh` / DeepSpeed             | port; passwordless SSH **between** nodes (not only laptop → rank 0)                                                          |
| GPUs per node                              | default 8                      | `nvidia-smi -L` if unknown                                                                                                   |
| Local disk path + shared-FS path           | `fio-scan`                     | discover with `df -hT`; ask if unclear                                                                                       |
| Advertised uni bandwidth / official TFLOPS | optional                       | else [accelerator tables](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/README.md) + `nvidia-smi` |

NCCL env is **not** a user question if you can exec: dump `env | grep -E '^NCCL_|^FI_|^AWS_OFI'` on rank 0 in this same preflight. Keep whatever the platform already sets for the fast path (`NCCL_IB_HCA`, `NCCL_NET_PLUGIN=ofi`, `NCCL_TUNER_PLUGIN=ofi`, EFA/OFI tunables, `NCCL_NVLS_ENABLE`, …). Do **not** overwrite a live platform value with the eval default. If NCCL is unset, **preset `NCCL_NVLS_ENABLE=2`**. Record the **final** env in the report (Environment + Network Introduction).

### Inter-node reachability (now, not a benchmark)

As soon as you have the hostfile/IPs, prove the nodes can see each other. **Do not** run `torch-dist-bench.py` or any GPU collective yet. This is seconds of TCP/SSH.

```bash
# 1. Launch node → every peer (BatchMode: fail fast if keys/port are wrong). Without -n, ssh reads the rest of the
#    hostfile from the loop's stdin and only the first peer is probed.
: > hostfile-map.txt
i=0
while read -r ip rest; do
  [[ "$ip" =~ ^# ]] || [[ -z "$ip" ]] && continue
  host=$(ssh -n -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new "$ip" hostname -s) \
    || { echo "FAIL: ssh $ip" >&2; break; }
  echo "node$i $ip $host" | tee -a hostfile-map.txt
  i=$((i + 1))
done < "$HOSTFILE"

# 2. Same via pdsh if that is the launcher
HOSTS=$(grep -v '^#' "$HOSTFILE" | grep -v '^$' | cut -d' ' -f1 | tr '\n' ',' | sed 's/,*$//g')
export PDSH_RCMD_TYPE=ssh   # every later pdsh call needs it too
pdsh -w "$HOSTS" hostname

# 3. Peer → peer TCP (rdzv will need this). From node 1, hit node 0's sshd at the port ssh uses, which `ssh -G`
#    reads from ~/.ssh/config: it's often not 22 in a container (7878 on one Kubernetes cluster).
MASTER_ADDR=$(grep -v '^#' "$HOSTFILE" | grep -v '^$' | head -1 | cut -d' ' -f1)
PEER=$(grep -v '^#' "$HOSTFILE" | grep -v '^$' | sed -n 2p | cut -d' ' -f1)
SSH_PORT=$(ssh -G "$MASTER_ADDR" | awk '$1 == "port" {print $2}')
ssh -n "${PEER:-$MASTER_ADDR}" "timeout 5 bash -c 'echo >/dev/tcp/$MASTER_ADDR/$SSH_PORT' && echo OPEN"
```

On Kubernetes, `ssh` may be `kubectl exec` per pod — still loop **every** sibling, not only the first (`-0`). Probe `sshd` listen port (`ss -ltnp | grep sshd`) and `ssh -p <port> <ip> hostname`. Do **not** assume port 22 or 2222. If PATH has a cluster SSH wrapper, use it; otherwise `ssh` / `pdsh`.

**Pass:** every hostname comes back, TCP to `MASTER_ADDR` is OPEN. **Fail:** ask the user (hostfile IPs, SSH port, keys, `pdsh` args) and wait. Do not start MAMF “in the meantime.”

Keep `hostfile-map.txt` in the eval directory, and copy it into `raw/` when you create the report: every `<host>(node<I>)` in the report comes from it, `node<I>` from its first column and `<host>` from its last. **Pass** also needs one row in it per hostfile line; a `FAIL: ssh <ip>` line names the peer the loop stopped at. With `kubectl exec` instead of `ssh`, write the same file from each pod's `hostname -s`, in hostfile order. On 1 node there is no hostfile, and the node is `node0`.

After the eval venv exists (loop step 2), add one more cheap check **before** GPU inter-node all-reduce: CPU-only, 1-process-per-node PyTorch `gloo` backend, `init_process_group` + `all_reduce` (see Network). Still not `torch-dist-bench.py`.

## Loop

0. **Preflight.** Fill the checklist. Ask for what’s missing (hostfile with peer IPs is required for **≥2 nodes**; 1 node needs no hostfile). **Stop until the user answers.** Then, if ≥2 nodes, run the SSH/TCP reachability probes. Do not start the isolated install or any long bench until that passes (1-node evals skip the probes and skip inter-node entirely).
1. Dump environment (below) into the report draft.
2. Isolated venv on the **newest** released torch (do **not** mutate or benchmark the node’s preinstalled Python).
3. If ≥2 nodes: CPU gloo 1-process-per-node across the hostfile (connectivity + rdzv). Fix `is_host` / `local_addr` here if needed. **Skip this step on 1 node.**
4. Connectivity: `torch-distributed-gpu-test.py` — one process per GPU; ranks must see each other and complete a collective. **The scope is whatever you launch it on**: across all nodes it is an inter-node check, on a single node it is only intra-node. With **≥2 nodes, run it across all of them** — a 1-node run does not test the fabric and must never be labelled inter-node.
5. **Compute:** the lemon GEMM first, on **every GPU of every given node** — it takes seconds and is a gate: a dead or grossly slow card stops the Compute phase there (see **Lemon GEMM**). Then `dcgmi diag -r 2` (hardware health), `mamf-finder-all-gpus.py` (one GPU at a time while every other GPU runs a continuous matmul), then, after 2 minutes with every GPU idle, `gpu-straggler-finder.py` on all of them at once.
6. **Network:** intra-node `torch-dist-bench.py --collectives all --separate-tables` — all-reduce bandwidth plus the per-call latency of every collective, in one run. Then, if **≥2 nodes**, the same on **all** of them (not a fixed 4). If **1 node**, stop after intra-node — no Inter-node heading.
7. **Storage:** `fio-scan` on local disk and on every unique shared partition (one per type and size, see **Storage**). With ≥2 nodes, on each partition scanned, a 1 GiB `dd` from node0 alone and then from every given node at once; skip the `dd` on 1 node.
8. Write `reports/<cluster>-<YYYY-MM-DD-HHMM>.md` next to this skill (or in the working directory), with `reports/raw-<name>/` beside it and **plots inlined**. Write it for a reader who knows nothing you learned on the node: every tool and term gets defined where it is first used. End with **## Findings**, split into **### Healthy subsystems**, **### Underperforming subsystems**, and **### Needs operator input** — the last for subsystems measured cleanly but with no known target to judge them against; keep only the headings that have bullets. Compute may use % of official TFLOPS. Intra-node all-reduce may use % of NVLink spec (note NVLS). Latency has no vendor target, so it goes under **Needs operator input** (see **Latency**). Storage: measured vs vendor advertised. **Never** flag multi-node all-reduce as “X% of NIC / rail spec.” Do not repeat the tables.
9. **Clear the scratch you created** (see **Leave the filesystems as you found them**) — before the summary, not after the user notices.
10. **Close-out:** **ask the user** to address each one, including every **Needs operator input** bullet — those are open questions addressed to them, and the eval stays unfinished until they supply the figures or say to drop it. For every gap: quote it, then give a **proposed plan of action** (commands, launcher, package, node count, or “accept and leave as-is”). Do not treat the eval as finished until they pick a plan, you execute it, or they explicitly accept the gap. If there are no Gaps sections, say the eval is complete.

Start the GPU performance pass, the lemon GEMM and `gpu-straggler-finder.py` only on a node with no other compute processes in `nvidia-smi`; abort if one appears. During `mamf-finder-all-gpus.py`, a python process on every GPU is the script's own matmul and is expected.

## Leave the filesystems as you found them

A storage eval writes enormous scratch files onto shared filesystems that belong to other people. **Clearing them is part of the eval, not an optional courtesy** — an eval that hands back a report and leaves 165 GiB across five production mounts has cost the cluster more than it measured. The user should never be the one to discover the leftovers.

What this eval creates, all of which has to go:

| what                                    | where                                              | size                                  |
| --------------------------------------- | -------------------------------------------------- | ------------------------------------- |
| `fio` work files                        | `<scan path>/fio-test/` on **every** mount scanned | ~33 GiB per mount per six-run scan    |
| concurrent-write poke files             | each share you wrote to, one per node + node0's alone | 1 GiB × (nodes + 1) per share      |
| `mamf-finder` / bench outputs, job logs | wherever you pointed them                          | small, but still yours                |
| the isolated venv, cloned repos         | `/tmp` or a share                                  | leave only if under node-local `/tmp` |

Rules:

- **Confine scratch to one timestamped directory per mount** — `<mount>/users/<user>/cluster-eval-<name>-<TS>/` — so clearing it is one move per mount and nothing of the user's sits inside the blast radius.
- **Keep the results, clear the data.** The JSON / summary files are the deliverable; copy them off first (see the raw-artifact rule), then handle the multi-GiB work files.
- **Move, never `rm`.** On the node: `mkdir -p <mount>/users/<user>/trash/<YYYYMMDD>-<topic> && mv <scratch> <that dir>/`. Node-local `/tmp` is the exception — it evaporates with the pod.
- **A move within one Lustre mount is a rename, but a move across mounts copies every byte.** Keep the trash directory **on the same mount** as the scratch, or a "cleanup" will take longer than the benchmark did.
- Moving to trash on the same filesystem **does not free the space**. Say so, give the exact paths, and let the user purge — they are the only one who may destroy data.
- **Report the cleanup in your summary**: what you moved, where to, and how much. Then the user can empty it in one command.

## Isolated install

**Always build a fresh venv on the newest released PyTorch that has a wheel for the driver's CUDA.** Never benchmark the node’s preinstalled env (conda, the image `python3`, a `dev` environment) — it is usually months behind. Torch version decides matmul kernels and NCCL version, so it changes MAMF and every `busbw` number. "It imports and the CUDA major matches" is **not** a reason to reuse it.

**Newest PyTorch, not newest Python: build the venv with the node's own Python if it is 3.12 or higher, else with 3.12.** The newest Python has no binary wheels for many packages yet, so never install one newer than the node has. If the node's Python is older than 3.12, install 3.12 the way the platform does (`apt-get install python3.12-venv`, `uv python install 3.12`, conda).

Find the newest PyTorch, do not assume a version from memory:

```bash
python3 -m venv "$EVAL_VENV"   # python3 -V >= 3.12, else python3.12; $HOME/venvs/cluster-eval, or shared FS so every node sees it
"$EVAL_VENV/bin/pip" install -U pip
"$EVAL_VENV/bin/pip" index versions torch --index-url https://download.pytorch.org/whl/cu130   # -> LATEST: x.y.z
"$EVAL_VENV/bin/pip" install torch numpy packaging nvidia-ml-py matplotlib \
  --index-url https://download.pytorch.org/whl/cu130
```

Pick the `cuNNN` index matching the driver (`nvidia-smi`); try the highest one the driver supports and fall back if it 404s. The newest PyTorch on PyPI or on a higher `cuNNN` index can be built for a newer CUDA than the driver runs — a driver for CUDA 13.0 found torch 2.15.0 built only for CUDA 13.2 and 13.4, and 2.14.1 the newest on `cu130` — so never install from those; the newest on the driver's index is the one. Record in the report the installed version **and** that it was the latest on that index at eval time. If the newest wheel cannot be used (driver too old, no matching index), say so in Gaps with the version you settled for — do not quietly benchmark an old torch.

**Verify each phase actually ran on it.** Every bench prints its stack (`- software: torch=…, nccl=…`); MAMF logs it too. A launcher can silently substitute its own interpreter — `deepspeed`/`srun` run whatever `python3` is on the remote PATH, so a run can come back on the node’s old torch even with the venv sourced. Grep the log before you copy a number into the report.

**`nvidia-ml-py` is required** (`import pynvml`) — `mamf-finder.py` refuses to start without it, because it needs the SM clock to confirm MAMF was a boost burst and the power draw to rank the scouted shapes. Do not pass `--telemetry off` to get around it; install the package. The pip name is `nvidia-ml-py`; the import is `pynvml`.

Host packages — install before the env dump, on **every** node you will measure. Prefix with `sudo` if the eval user is not root. Empty `ibstat` on EFA is OK; still install `infiniband-diags` and confirm fabric with `rdma link` / `/sys/class/infiniband`.

Debian/Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y infiniband-diags fio iproute2 rdma-core pdsh
# DCGM is not in Ubuntu — NVIDIA CUDA repo (after cuda-keyring). Match CUDA major from nvidia-smi:
#   CUDA_VERSION=$(nvidia-smi | sed -E -n 's/.*CUDA Version: ([0-9]+)[.].*/\1/p')
#   apt-get install -y --install-recommends datacenter-gpu-manager-4-cuda${CUDA_VERSION}
```

RHEL/Rocky:

```bash
sudo dnf install -y infiniband-diags fio iproute rdma-core pdsh
# DCGM: NVIDIA CUDA repo, then datacenter-gpu-manager-4-cuda${CUDA_VERSION} (CUDA major from nvidia-smi).
```

| binary   | package                                                         | why                                                                                                            |
| -------- | --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `ibstat` | `infiniband-diags`                                              | InfiniBand/RoCE NIC (channel adapter) state, rate, LID. **Empty on EFA is OK** — then `rdma link` + sysfs rate |
| `fio`    | `fio`                                                           | `fio-scan`                                                                                                     |
| `rdma`   | `iproute2` (+ `rdma-core`)                                      | `rdma link`                                                                                                    |
| `pdsh`   | `pdsh`                                                          | DeepSpeed multi-node launcher / torchrun helper                                                                |
| `dcgmi`  | `datacenter-gpu-manager-4-cudaN` (NVIDIA CUDA repo, not distro) | `dcgmi diag -r 2` hardware health                                                                              |

If apt/dnf is blocked, record that in Gaps and fall back to `/sys/class/infiniband/*/ports/*/rate` + `nvidia-smi topo` — that is a last resort, not the default.

The five scripts from **Scripts (fetch)** must sit in **one directory** on a filesystem every node can read (shared FS), so `fio-scan` can call `./fio-json-extract.py`.

## Environment dump

Capture into the report (datetime in the report's time zone, hostnames):

```bash
date -u +%FT%TZ
hostname -s; uname -a
nproc; grep MemTotal /proc/meminfo
. /etc/os-release; echo "$PRETTY_NAME"
"$PY" -V
"$PY" -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.nccl.version())"
nvidia-smi --query-gpu=index,name,uuid,serial,pci.bus_id,driver_version,memory.total,power.limit,power.max_limit,clocks.max.sm,clocks.sm,power.draw,pcie.link.gen.current,pcie.link.gen.max --format=csv
nvidia-smi topo -m
# fabric (install infiniband-diags / iproute2 first — see Isolated install):
ibstat
rdma link
ls /sys/class/infiniband
cat /sys/class/infiniband/*/ports/1/rate
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv
sha256sum mamf-finder.py mamf-finder-all-gpus.py torch-dist-bench.py gpu-straggler-finder.py
```

On ≥2 nodes, also save every node's GPU identities, since the UUIDs and serials that the lemon gate and **Findings** ask for may be on any node: `pdsh -w "$HOSTS" "nvidia-smi --query-gpu=index,uuid,serial,pci.bus_id --format=csv,noheader" > gpus.csv`, copied into `raw/` with the report. On 1 node the `--query-gpu` line above has them.

Classify intra-node (NVLink vs PCIe from `topo`) and inter-node (InfiniBand / RoCE / EFA / Ethernet). HBM = `memory.total` reported in **GiB** (not MiB). SM clocks in the Environment GPU table: **boost** = `clocks.max.sm` (spec / short burst); **saturated** = SM clock at ~TDP under a dense GEMM (a range is fine; not a single nvidia-smi field); **parked idle** = `clocks.sm` when no compute. Do not label boost as “loaded” — a power-saturated GEMM sits well below boost. Do **not** name MAMF/MSMF in Environment — those terms start in Compute. Write the dump as **five tables** in the report: Host / GPU / Fabric / Filesystems / Software. Never one mega-table. **Software:** OS is **not** here — it is Host row 1. Python version + torch/NCCL. **Never** the eval venv path (temporal artifact). Tool versions (`mamf-finder`, …) belong under Compute.

## Compute

### Introduction

Open Compute with the **GPUs that are here** (name, count per node, SM count, HBM GiB, TDP, official BF16 TFLOPS + [table](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/README.md#tflops-comparison-table)). Do not start with DCGM or a FLOPS table. Then introduce the four passes as a **numbered list** (`1.` … `4.`), each with the tool: `dcgmi diag -r 2` (NVIDIA's health check), [`mamf-finder.py`](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/benchmarks/mamf-finder.py) (shape search → **MAMF** burst *and* **MSMF** sustainable TFLOPS, both vs spec — do not describe it as burst-only), lemon GEMM (inline script), [`gpu-straggler-finder.py`](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/benchmarks/gpu-straggler-finder.py) (4 workloads on every GPU at once, each GPU against its node's median GPU and each node against the median node). Headings below are **Hardware health**, **GPU performance benchmarks**, **Find underperforming GPUs** — not the tool nicknames.

### Hardware health

Hardware health, not a FLOPS number. `-r 1` is seconds of software/deployment only — skip it as a named step. **`-r 2`** is the light hardware suite (GPU memory, memory bandwidth, PCIe/NVLink; NVIDIA: ≲10.5 min on 8 H200s, 13 min on 8 B200s). `-r 3` is a longer stress test, optional.

Needs `nv-hostengine` running and usually `CAP_SYS_ADMIN` (these trial pods are typically root with it). Start the host engine with `sudo` when the eval user is not root: started as the user it exits, and `dcgmi diag` then fails with `Unable to connect to host engine` and exit 253. If install or diag fails, **Gaps** and continue with MAMF and the straggler pass.

```bash
pgrep -x nv-hostengine >/dev/null || sudo -n nv-hostengine   # drop sudo if root
sleep 3
dcgmi discovery -l
dcgmi diag -r 2 | tee dcgm-diag-r2.txt
```

One node is the minimum; extra nodes in parallel if you already have them. Report **Pass** or **Fail** per plugin and per GPU. If everything passed, write **Pass** — do not add **No Fail**. Copy the log off immediately.

### GPU performance benchmarks

`mamf-finder-all-gpus.py` — one GPU at a time, while every other GPU of the node runs a continuous bf16 matmul that holds it at its power limit, as in a real workload:

```bash
OUT_DIR=mamf "$PY" mamf-finder-all-gpus.py --dtype bfloat16 > mamf-all.console 2>&1
```

On 8 H200s it took about 4.5 minutes; on 8 B200s, 9.5 minutes, every node at once.

1. GPU0 runs the full `--search auto`: MAMF and MSMF with their shapes (`mamf/gpu0.txt`, whose `Tried N shapes` line gives the search size).
2. Each other GPU in turn measures GPU0's MAMF and MSMF shapes only (`mamf/gpu<N>.txt`); that is much shorter than a search.
3. `mamf/summary.txt` lists each GPU's MAMF and MSMF with shapes, W, MHz and temperature, then for each headline the node figure (the median GPU), the slowest GPU, which synchronous work runs at, and the spread across GPUs.

Every GPU's MAMF and MSMF are measured on the same 2 shapes, so the numbers compare directly; only GPU0's are search results. Compare MAMF and MSMF to official BF16 TFLOPS. A rare GPU's MAMF on GPU0's shape can lose to another shape on that GPU (the summary then shows a different MAMF shape for it); report it as printed.

A `hotter than the coolest GPU` warning names a GPU whose MSMF window ran 15°C or more hotter than the node's coolest GPU. It is a fact about the node's cooling, not a fault; report it in a sentence below that node's table. If it names the same GPU index on every node, say so: that is the chassis' airflow, not a card — on 4 nodes of 8× B200 it named GPU0 on all 4, and the even-numbered GPUs ran about 10°C hotter than the odd ones in every pass.

**One pass cannot call a GPU slow, and a few percent is not slow.** Healthy GPUs differ from chip to chip: on healthy 8x H200 and 8x B200 nodes the slowest GPU's MSMF sat 0.9–1.7% below its node median, and a single GPU's MSMF moves by up to 2.2% between passes on top of that, so a healthy GPU can read about 4% below the median in one pass. A GPU is a candidate only if it is more than 5% below the node median MSMF, well beyond that spread. Then rerun the pass twice with the shape pinned — `--m M --n N --k K` from the summary's MSMF shape, `OUT_DIR=mamf-rep2` and `mamf-rep3`, about 3.5 minutes each on 8 H200s. A GPU is under-performing only if it is more than 5% below its node median in **all three** passes; report it in **Findings** as described there. If the dip did not repeat, say so in one sentence with the three numbers, so the reader does not wonder about the first table.

- A non-zero exit means a GPU's run failed: read `mamf/gpu<N>.err`, rerun or put it in **Gaps**.
- A `saw idle siblings` warning in `summary.txt` means a sibling's matmul stopped during that GPU's measurement, so its MSMF reads high. Rerun; if it repeats, put it in **Gaps**.
- Do not substitute plain `mamf-finder.py` per GPU with the others idle: that MSMF is a single-GPU upper bound, and its log ends with a note saying so.

### Find underperforming GPUs

An under-performing GPU, a **straggler**, works correctly but runs slower than its peers, and in any synchronous job — training, or inference with tensor, expert or pipeline parallelism — every GPU waits for it at each collective. It passes DCGM, and the GPU performance pass can miss it: a B200 that made a training step 3–5% slower than its 7 siblings, only some of the time, had a MAMF and MSMF within 1–2% of its node's median, as both report a GPU's best stretch. Two passes, both on every GPU of every given node, **not** MAMF:

1. **Lemon GEMM** — seconds, run first in the Compute phase, before DCGM and the GPU performance pass: a gate for a dead or grossly slow card. Its section stays here in the report.
2. **[`gpu-straggler-finder.py`](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/benchmarks/gpu-straggler-finder.py)** — the verdict, run last: a straggler is a few percent slow, only some of the time, which one fixed matmul with idle siblings doesn't show.

The 3 passes overlap but answer different questions: the GPU performance pass measures whether the GPU model delivers its spec, in absolute TFLOPS; the straggler finder compares each GPU with its peers, so it would call 32 GPUs that are all 20% under spec healthy; the lemon GEMM finds in seconds what would otherwise surface only 15 minutes later, as a failed MAMF or finder run.

A GPU goes to **Findings** when any of the 3 confirms it: the lemon GEMM's gate (dead, or more than 5% below its node's median in 3 runs), the GPU performance pass (more than 5% below its node's median MSMF in 3 passes), or the finder (flagged in all 4 runs, in the same workload). The bars differ because the noise does: a single MSMF or lemon number moves by a few percent between passes, while the finder divides each GPU's time by its node's median GPU's in the same 2-second window, which cancels what the node's GPUs share, and on 4 nodes of 8× B200 no healthy GPU's 90th percentile step window went above 1.9% in 5 runs - the finder also requires a modified z-score above 3.5, which such a GPU doesn't reach. A GPU that two passes confirm gets one bullet naming both. One that only the finder flags is still under-performing: that is the straggler the other two miss.

Open with **TLDR: no under-performing or dead GPUs**, or name each one: **TLDR: under-performing: `<host>(node<I>):gpu<N>`**, `dead: <host>(node<I>):gpu<N>`, `slow node: <host>(node<I>)`.

#### Lemon GEMM

Same large `matmul` on every GPU, siblings idle. Run it on 1 node or 16.

Fixed **16384³ bf16**. Compute SM coverage from **this** GPU’s SM count (`torch.cuda.get_device_properties(0).multi_processor_count`) — never reuse an SM count from another GPU family. At 128×256 output tiles that is **8192** CTAs (`16384/128 × 16384/256`; a CTA is one CUDA thread block). Full waves = `8192 // SMs`, tail = `8192 % SMs`, wave efficiency = `1 - tail/8192`. Example only: 132 SMs → 62 waves + 8 CTA tail (**99.9%**).

Report it in one sentence when nothing trips the gate: the median of all the GPUs' TFLOPS, the range, and the lowest GPU with how far below its node's median it is. A GPU that trips the gate gets a row with its TFLOPS in each of its 3 runs and its node's median. The per-GPU scores stay in `raw/lemon.console`. A card more than 5% below its node median goes to **The gate** below, which decides it; the min of a tight pack is noise.

**The gate.** A GPU whose run errors, hangs or prints `inf`/`nan` is dead. One more than 5% below its node's median GPU is grossly slow if it stays there when the script is run twice more, which takes seconds. Either one stops the Compute phase: put it under **Underperforming subsystems**, and tell the user which `<host>(node<I>):gpu<N>` it is, with its UUID and serial from the environment dump, so the node can be fixed or replaced before the long passes. Continue with Network and Storage, which don't depend on it. If nothing trips the gate, go on to DCGM.

Passing the gate doesn't clear a GPU: on 4 nodes of 8× B200 the GPU the finder then flagged in all 4 runs read 4.4% below its node's median here, the lowest of the 32 but under the 5% bar. Name the lowest GPU in that sentence, with how far below its node's median it is, so the reader can match it against the finder's verdict.

#### Stragglers

`gpu-straggler-finder.py` runs 4 workloads on every GPU of every node at once, with no communication between the GPUs while it times them: a Llama-3.1-8B training step (its 32 decoder layers on one 8192-token sequence, forward and backward, each step followed by a 100 ms idle) for 60 s, then for 30 s each a back-to-back matmul (8192³ bf16), a device-to-device `cudaMemcpy` and a kernel adding 2 HBM buffers into a third. It cuts each workload into 2 s windows and divides every GPU's time by its node's median GPU's in the same window, and compares each node's median GPU with the median node's. `pip install nvidia-ml-py` in the eval venv first, for its clock, power and temperature columns; without it only the timings are reported.

On ≥2 nodes, set up first, from the shared directory the scripts are in. `MASTER_ADDR`, `HOSTS` and `HOSTFILE` are the ones from **Inter-node reachability**; in a new shell, derive them again the same way.

```bash
export PDSH_RCMD_TYPE=ssh
EVAL=$(pwd -P)
NGPU=$(nvidia-smi -L | wc -l)
NNODES=$(grep -v '^#' "$HOSTFILE" | grep -vc '^$')
# gloo's interface: the one that carries each node's hostfile IP. If its name differs between nodes, ask the user.
pdsh -w "$HOSTS" "ip -o -4 addr show"
[ -n "$GLOO_IFACE" ] || echo "STOP: set GLOO_IFACE to the interface the line above shows the hostfile IPs on"
# --hostfile-to-sort writes <file>-fast-to-slow next to the file it is given: give it a copy, never the platform's hostfile
cp "$HOSTFILE" "$EVAL/hostfile"
```

Don't go on until `GLOO_IFACE` is set. Then run one of the 2 forms below. Each starts with 2 minutes with every GPU idle, right before the launch, as a straggler can be slow only in the first 20-25 s after the load starts from idle:

```bash
# 1 node; --unflagged-rows -1 keeps every GPU's row in the console, the raw record the report links to.
# Without torchrun it exits 3 if it flagged a GPU and 2 on inf/NaN: that is its verdict, not a failed run.
sleep 120
"$PY" gpu-straggler-finder.py --unflagged-rows -1 --json straggler.json > straggler.console 2>&1
```

```bash
# ≥2 nodes: the same command on every node at once, under torchrun; it gathers the results on rank 0 over gloo
sleep 120
pdsh -w "$HOSTS" "cd $EVAL && GLOO_SOCKET_IFNAME=$GLOO_IFACE $PY -m torch.distributed.run --nnodes $NNODES \
  --nproc_per_node $NGPU --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:29500 --max_restarts 0 \
  gpu-straggler-finder.py --unflagged-rows -1 --json $EVAL/straggler.json --hostfile-to-sort $EVAL/hostfile" \
  > straggler.console 2>&1
```

If any node's own hostname resolves to `127.0.1.1` (`getent hosts $(hostname)` on it), add `--rdzv_conf is_host=… --local_addr …` as in **`is_host` / `local_addr`**.

Each verdict line names a GPU as `<host> GPU <N>`, and the table rows give the host and the index in their own columns: `<host>` is the node's hostname, to match against `hostfile-map.txt` up to the first `.`, which gives its `node<I>`, and `<N>` is the `nvidia-smi` index, since the finder matches GPUs by UUID. After a run on ≥2 nodes in which no GPU returned inf/NaN, it ends with the nodes ranked by the pace of a job that runs their GPUs in lockstep: the nodes that passed, fastest first, then every node with a flagged GPU, or flagged itself. It writes the hostfile's lines in that order to `$EVAL/hostfile-fast-to-slow`, so its first N lines are the N fastest nodes that passed, if at least N passed. Copy that file into `raw/` with `straggler.json` and `straggler.console`. Two messages mean that the hostfile and the nodes the finder measured disagree - a node missing from the run, or a hostfile IP that no node owns: `left out N line(s) of <file> that name no node measured, ...` and `no line of <file> names <host>, ...`. Put each in **Gaps**. The stall lines' "left out of that node's comparisons" is not one of them.

One run took 3 min 10 s on 4 nodes of 8× B200. Under torchrun its exit code is always 0; the verdict is the framed banner in the report it prints: `No Stragglers Found`, or `Stragglers Found:` with the number of flagged GPUs and slow nodes, followed by one line per flagged GPU or node, with what it was flagged for, its numbers, and its UUID, serial and PCI bus id. A GPU is flagged when its 90th percentile window in a workload is above a modified z-score of 3.5 among all the GPUs and more than 1% slower than its node's median GPU, (**slow**), or when its 90th percentile window is more than 1.25% slower than its own median window, also above a modified z-score of 3.5 (**unsteady**: slow only some of the time); a node when its median GPU is more than 3% slower than the median node's.

**One run cannot call a GPU under-performing either.** If the first run flags anything, run it again with `--repeat 3`, 12 minutes on 4 nodes of 8× B200, writing its own files so that the first run's survive: the verdict needs all 4 runs. `--repeat-idle` idles only between its runs, so idle before the first one too. With ≥2 nodes:

```bash
sleep 120
pdsh -w "$HOSTS" "cd $EVAL && GLOO_SOCKET_IFNAME=$GLOO_IFACE $PY -m torch.distributed.run --nnodes $NNODES \
  --nproc_per_node $NGPU --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:29500 --max_restarts 0 \
  gpu-straggler-finder.py --unflagged-rows -1 --repeat 3 --repeat-idle 120 --json $EVAL/straggler-repeat3.json \
  --hostfile-to-sort $EVAL/hostfile" > straggler-repeat3.console 2>&1
```

On 1 node: `sleep 120; "$PY" gpu-straggler-finder.py --unflagged-rows -1 --repeat 3 --repeat-idle 120 --json straggler-repeat3.json > straggler-repeat3.console 2>&1`. Copy both files into `raw/`. Instead of a banner per run it prints one, `Stragglers Found in 3 Runs:` with the GPUs flagged in all 3 and in some, or `No Stragglers Found in 3 Runs`. Its "in some" also counts the GPUs that were only slower than in their own fastest run, and its "all 3" leaves out the first run, so take the 4-run verdict from `straggler.json` and `straggler-repeat3.json` together. Its sorted hostfile ranks each node by its slowest of the 3 repeat runs and overwrites `$EVAL/hostfile-fast-to-slow`; copy it over the one in `raw/`. A GPU or node is under-performing only if it was flagged in **all 4** runs, in the same workload; report it in **Findings** as described there. One flagged in some of the runs goes in this section only, in one sentence with the runs it was flagged in and its numbers in each. On 4 nodes of 8× B200 a GPU 3.3% slow in the matmul alone, at the lowest matmul clock of the 32 under the power cap and normal in the step, was flagged in 2 runs of 4.

- A run that stops early or prints `inf or NaN` is a dead or broken card: name it from the console and put it under **Underperforming subsystems**.
- `stall` lines name windows in which all of a node's GPUs slowed at once, which the comparisons leave out; report them per node, as they slow a job too, but they are not a GPU's fault.
- The `--repeat` summary also lists a GPU whose median in the step, the copy or the stream was more than 1% slower in a run than in its own fastest run; not in the matmul, which moved by up to 1.2% between runs on healthy GPUs. That is not one of the 4 runs' flags: one sentence in this section, with the runs, and nothing in **Findings** unless the GPU was flagged in all 4.
- On 1 node the finder can't compare nodes, and on 2 it can tell them apart but not which one is normal: it flags a node whose median GPU's 90th percentile window is more than 3% slower than the faster node's. Say so in one sentence; if the user has the `--json` of a run on known-good nodes of the same GPU, rerun with `--baseline <file>` to compare against them.
- A GPU flagged in all 4 runs: record its node's SM clock with Nsight Systems, which samples it far more often than the finder's 10 ms NVML samples, between which a dip can hide. On that node alone:

  ```bash
  report=straggler-$(hostname -s)
  sleep 120
  nsys profile --trace=cuda,nvtx --gpu-metrics-devices=all -o "$report" "$PY" gpu-straggler-finder.py \
    --unflagged-rows -1 > "$report.console" 2>&1
  nsys export --type sqlite --output "$report.sqlite" "$report.nsys-rep"
  ```

  The `GPU_METRICS` clock values in the `.sqlite` are in Hz despite the `[MHz]` in their name. Use this run for the clocks only, not for its verdict: nsys slows the GPUs unevenly, and on a node of 8× H200 that passed 4 runs without it, its step was 2% slower and 3 GPUs were flagged slow in it at +1.3-1.7%. Copy the `.nsys-rep`, the `.sqlite` and the `.console` into `raw/`; for 8 GPUs they were 0.6 GB and 19 GB, and the export took 6 minutes. If `nsys` says the GPU performance counters are in use, stop `nv-hostengine`, which DCGM may have left running, and run it again. If `nsys` isn't installed, put it in **Gaps**.

## Network

Ask which NCCL/platform env vars to set. **Dump first** (`env | grep -E '^NCCL_|^FI_|^AWS_OFI'`). Keep platform values. Default **only if unset**: `NCCL_NVLS_ENABLE=2`. If `NCCL_TUNER_PLUGIN=ofi` (or similar), a **platform tuner** — not NCCL’s built-in `tuning.cc` — picks the algorithm; confirm with `NCCL_DEBUG`. `pdsh` does **not** copy the launcher’s environment — put overrides in `~/.deepspeed_env` **and** `source` them in the remote command, or embed `VAR=val` in the pdsh `cmd`. `--venv_script run-env.sh` is the DeepSpeed equivalent. DeepSpeed can drop exports from `--venv_script`; `~/.deepspeed_env` (one `KEY=VALUE` per line, in the launch cwd) is the reliable file. **`--venv_script` does not switch the Python interpreter** — DeepSpeed still launches whatever `python3` is on the runner PATH (often the node’s old install). Put the eval venv **first** on `PATH` inside `run-env.sh` *and* invoke `deepspeed` from that venv (`$EVAL_VENV/bin/deepspeed` or `PATH=$EVAL_VENV/bin:$PATH deepspeed …`). Confirm the table’s `torch=` / `nccl=` line matches the isolated install.

Connectivity first (`torch-distributed-gpu-test.py` — ranks must init NCCL and finish a collective). **The heading and TOC label must match the ranks you actually launched**: `Inter-node connectivity` only when it ran across every node, `Intra-node connectivity` for a single-node run. On ≥2 nodes use the same launcher as inter-node all-reduce (`deepspeed -H`, `pdsh`, …) with one rank per GPU; the snippet below is the 1-node form.

```bash
"$PY" -m torch.distributed.run --nproc_per_node="$NGPU" --rdzv_endpoint localhost:6000 --rdzv_backend c10d \
  torch-distributed-gpu-test.py
```

Intra-node (header of [`torch-dist-bench.py`](https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/torch-dist-bench.py)):

```bash
"$PY" -u -m torch.distributed.run --nproc_per_node="$NGPU" --rdzv_endpoint localhost:6000 --rdzv_backend c10d \
  torch-dist-bench.py --collectives all --separate-tables
```

`--collectives all` times all nine collectives (`all_reduce` first), each from 8 B to 16 GiB; on one 8x B200 node they take under 2 minutes, on 4 nodes about 3. The `all_reduce` table is the all-reduce benchmark below; the per-call latency of all nine is the latency benchmark (see **Latency**).

Inter-node (**≥2 nodes**, **all** of them). Let `NNODES` = hostfile length, `NGPU` = GPUs per node, ranks = `NNODES * NGPU`. **1 node: omit this whole subsection** — no heading, no “Skipped: 1 node”. **Do not discover a missing hostfile here** — that was the preflight gate. Use whatever launcher this cluster already has (`deepspeed -H <hostfile>`, `pdsh`, `srun`, or a site SSH wrapper). Reachability (SSH/TCP) already passed; this section is the GPU `torch-dist-bench.py --collectives all --separate-tables` sweep only. Its results go in the same `### All-reduce benchmark` and `### Latency benchmark` sections as the intra-node ones (ranks go in the TOC annotation, not the heading).

Hostfile: skip `#` comments. `MASTER_ADDR` = first **data** line, not a comment. If the platform already ships a hostfile, **read it, never edit it**.

**1. DeepSpeed** (from the shared cwd that contains the bench):

```bash
# run-env.sh: source the eval venv; re-export the NCCL values you dumped (do not invent 2 if the platform set 1)
time deepspeed -H "$HOSTFILE" --venv_script run-env.sh torch-dist-bench.py --collectives all --separate-tables \
  |& tee torch-dist-bench-deepspeed.log
```

Requires passwordless SSH from the launch node to every hostfile IP. Port is cluster-specific (a wrapper on PATH may already add `-p`). If `ssh` hits the **machine** sshd (not the container), start `sshd` in the container on an unused port, put the launch key in each container `authorized_keys`, and set `PDSH_SSH_ARGS_APPEND="-p <port> -o StrictHostKeyChecking=accept-new"`. Ubuntu `pdsh` (used by DeepSpeed) may refuse to load modules when `/usr/lib` is not root-owned on overlay — `sudo chown root:root /usr/lib` on the launch node if you are not root.

`pip install matplotlib` in the eval venv on **shared FS** so **every** node can plot; otherwise only the intra-node (single-node) plot appears.

**2. pdsh + torch.distributed.run** (pdsh does not forward env):

```bash
GPUS_PER_NODE=8
NNODES=$(grep -v '^#' "$HOSTFILE" | grep -vc '^$')
MASTER_ADDR=$(grep -v '^#' "$HOSTFILE" | grep -v '^$' | head -1 | cut -d' ' -f1)
MASTER_PORT=6002
HOSTS=$(grep -v '^#' "$HOSTFILE" | grep -v '^$' | cut -d' ' -f1 | tr '\n' ',' | sed 's/,*$//g')
cwd=$(pwd)

# If getent hostname is 127.0.1.1, add to the remote cmd:
#   MYIP=\$(hostname -I | awk '{print \$1}');
#   IS_HOST=\$([ \"\$MYIP\" = \"$MASTER_ADDR\" ] && echo 1 || echo 0);
#   --rdzv_conf is_host=\$IS_HOST --local_addr \$MYIP
cmd="source $cwd/run-env.sh; \
python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --role \$(hostname -s): \
    --tee 3 \
    $cwd/torch-dist-bench.py --collectives all --separate-tables"

PDSH_RCMD_TYPE=ssh PDSH_SSH_ARGS_APPEND="-o StrictHostKeyChecking=accept-new -o BatchMode=yes" \
  pdsh -w $HOSTS $cmd |& tee torch-dist-bench-torchrun.log
```

`--role \$(hostname -s)` must expand **on the remote**, not on the launch node. `MASTER_ADDR` must be an IP other nodes can **TCP** to (rdzv).

#### `is_host` / `local_addr`: only when hostname ≠ routable IP

Probe first: `getent hosts "$(hostname)"` and `_matches_machine_hostname(MASTER_ADDR)` on the endpoint node. If hostname already maps to the pod/node IP, **omit** `--rdzv_conf is_host` and `--local_addr`. Add them only when hostname is `127.0.0.1` / `127.0.1.1` or `_matches_machine_hostname` is False on the node that owns the endpoint.

When needed, two independent failures look like a `TCPStore` timeout:

1. **Nobody hosts the store.** With no `is_host`, each agent calls `_matches_machine_hostname(endpoint_host)` ([`elastic/rendezvous/utils.py`](https://github.com/pytorch/pytorch/blob/main/torch/distributed/elastic/rendezvous/utils.py)), which only compares the endpoint against IPs from resolving `socket.gethostname()`. When the hostname maps to `127.0.1.1`, that is **False on the node that owns the endpoint IP**, so every agent starts as a *client*, nothing ever listens, and all nodes — the master included — time out. `is_host=1` on the endpoint node forces the `TCPStore` server (it binds `*:PORT`).
2. **Workers get an unresolvable master.** Rendezvous then advertises the host agent's *hostname* as `MASTER_ADDR`. Workers that cannot resolve it retry forever with `[c10d] The IPv6 network addresses of (<hostname>, <port>) cannot be retrieved (gai error: -2 - Name or service not known)`. `--local_addr $MYIP` makes each agent advertise its IP instead.

Diagnose in this order, do **not** guess at the fabric — TCP already works if DeepSpeed runs:

```bash
# 1. does the endpoint node think it is the host?
python -c "from torch.distributed.elastic.rendezvous.utils import _matches_machine_hostname as m; print(m('$MASTER_ADDR'))"
# 2. with is_host=1, does the store bind?           -> expect *:PORT
ss -ltnp | grep "$MASTER_PORT"
# 3. can a worker reach it?                          -> expect OPEN
timeout 5 bash -c "echo > /dev/tcp/$MASTER_ADDR/$MASTER_PORT" && echo OPEN
# 4. is the advertised master resolvable everywhere?
getent hosts "$(hostname)"      # 127.0.1.1 => you need --local_addr
```

Verify with a **CPU-only, 1-proc-per-node** `gloo` script (`init_process_group` + `all_reduce`) before spending GPUs; it isolates rendezvous from NCCL. `--rdzv_conf timeout=60` keeps a failed attempt short instead of the 600 s default. Symptom cheat sheet: *nothing listening anywhere* → problem 1; *store listening, workers log `gai error: -2`* → problem 2.

### Transport detection

Do this **before** quoting any all-reduce number. `busbw` is already unidirectional ([directionality](https://github.com/stas00/ml-engineering/blob/master/network/README.md#unidirectional-vs-bidirectional-duplex)). Tables: [intra-node](https://github.com/stas00/ml-engineering/blob/master/network/README.md#all-to-all-bandwidth), [NVLink gens](https://github.com/stas00/ml-engineering/blob/master/network/README.md#nvlink), [inter-node / adapters](https://github.com/stas00/ml-engineering/blob/master/network/README.md#inter-node-networking).

**Intra-node** — the only place “% of spec” is a wire comparison (NVLink/PCIe). NVLS can push ring-formula `busbw` past 100% of that spec; say so.

```bash
nvidia-smi topo -m          # NV18 = 18 NVLink bonds; PHB = PCIe host bridge (no NVLink)
nvidia-smi nvlink -s        # per-link GBps; ×0.94 ≈ advertised (encoding)
```

Map NVLink gen from per-link payload rate (or GPU family): ~53 GBps/link → **NVLink 5** (900 GBps unidirectional, B200-class); ~26.5 → **NVLink 4** (450, H100/H200-class); slower ~12.5-class is older NVLink ([table](https://github.com/stas00/ml-engineering/blob/master/network/README.md#nvlink)). `NV#` in topo is bonded link count. No NV* → PCIe; use `pcie.link.gen.max` (Gen5 x16 ≈ 63 GBps uni). Intra-node all-reduce % = `peak_busbw / spec_uni * 100` (NVLS can exceed 100% of the wire formula — note that).

**Inter-node**

```bash
ls /sys/class/infiniband/                 # mlx5_* IB/RoCE; rdmap* EFA
cat /sys/class/infiniband/*/ports/1/rate  # e.g. 200 Gb/sec (EFA) or 800 Gb/sec (CX-8)
rdma link
ibstat   # may print **nothing** on EFA — that is not “no fabric”; use rdma link + sysfs
nvidia-smi topo -m                        # NIC legend when present
```

Identify adapter (AWS EFA + `fi_info -p efa`, NVIDIA ConnectX-7 NDR, ConnectX-8 XDR, …) from rate + name. Rail-optimized fabrics often put **one NIC per GPU**; count `ls /sys/class/infiniband | wc -l` / GPU count and record as **inventory**, not a busbw denominator. “Rail” here means one NIC lane per GPU, not a score.

**Do not score multi-node all-reduce against the NIC spec — not in the Network section, not in Findings.**

`NNODES ≥ 2` `torch-dist-bench.py` `busbw` is **always** mixed: intra-node NVLink + inter-node NIC + (usually) NVLS/SHARP. Advertised per-GPU IB/RoCE/EFA GBps is **inventory**. It cannot be measured in isolation with this collective, NVLS on or off. NVLS is in-network all-reduce on the NVSwitch (~30% intra / ~25% inter vs ring `busbw`) and does **not** help other collectives.

Forbidden (do not score multi-node all-reduce as a fraction of NIC rate — that mixed NVLink+NIC+NVLS number is not “% of rail”):

- `busbw / 100 GBps` (or any NIC uni rate)
- `busbw * (k-1)/(n-1)` as “GBps on the wire” or “% of rail” — that factor is ring-traffic bookkeeping in [why inter-node busbw is not the wire](https://github.com/stas00/ml-engineering/blob/master/network/README.md#real-network-throughput), not an eval score, and it is wrong with NVLS (the default preset)

Quote instead: peak **busbw**, NCCL path (IB vs Socket; NVLS yes/no), NIC inventory on the Environment / transport line, and optionally multi-node busbw vs **1-node** all-reduce at the **same payload** (scale-out tax). Record `NCCL_NVLS_ENABLE` and other NCCL env.

Report template must include: transport + version, NIC inventory GBps, measured peak busbw, NCCL path — **not** “% of NIC spec.”

### Latency

The same `--collectives all --separate-tables` run gives the per-call latency of every collective: one table per collective with the latency median, p99 (99th percentile) and mean per payload, each call's time being the slowest rank's; then a summary table and `latency-collectives-<host>-<ranks>.png`, which plots all nine. Copy that plot off with the others.

**Do not let all-reduce stand for latency.** NCCL gives only all-reduce its tree algorithm, whose latency grows with the log of the node count while the rings of the other collectives grow linearly, and with NVLS or InfiniBand SHARP a small all-reduce can be reduced inside the switch, so on some clusters it has the lowest small-payload latency of all. On others it has nearly the highest: on 4 nodes of 8× B200 with EFA and `NCCL_NVLS_ENABLE=1`, an 8 B all-reduce took 23.5 µs within a node, the highest of the nine, against `broadcast`'s 11.4 µs, and 76.0 µs over the 4 nodes, against `broadcast`'s 24.5 µs. Which collective is fastest is a property of the cluster, so never assume it in the report: report all nine, and say in the report what the main ones show — [which collective to measure latency with](https://github.com/stas00/ml-engineering/blob/master/network/README.md#which-collective-to-measure-latency-with):

- `batch_isend_irecv` — point-to-point, how much latency the NIC and switches add without a collective algorithm in the way (pipeline parallelism, KV-cache transfer).
- `all_to_all` — every GPU sends to every other at once, which exposes oversubscription and incast (MoE expert-parallel dispatch and combine).
- `all_reduce` — the best case (tensor-parallel inference within a node, where it really is the workload).
- `all_gather` / `reduce_scatter` — ring latency, growing linearly with the rank count (FSDP/ZeRO training at scale).

PyTorch runs `batch_isend_irecv` through its point-to-point path and `scatter` as separate NCCL sends, while `nccl-tests` makes one NCCL call for each, so these two can read higher than `nccl-tests` would on the same cluster. Say so next to their numbers.

One table for both scopes: `| collective | intra-node<br>median (µs) | inter-node<br>median (µs) | inter/intra<br>× |` at the smallest payload, named in the row of a collective that splits it (see below) — `<br>` inside the single header row keeps it narrow enough for PDF, which can't scroll sideways. At that payload a call's time is nearly all latency: on 8x H200 every all-reduce up to 512 KiB took 15–23 µs. **`inter/intra ×`** is the inter-node median ÷ the intra-node median of the same collective at the same payload — how many times longer a call takes once it leaves the node. Inline the `latency-collectives-…` plot of each scope.

`all_gather`, `reduce_scatter`, `all_to_all`, `gather` and `scatter` split the payload across the ranks, so their tables start above 8 B — at 128 B on 8 ranks. Use the smallest payload each one ran in the 8 B columns and name it in the row, e.g. `all_gather` (128 B), rather than leaving the cell empty.

**Latency has no target.** No vendor publishes collective latency figures, so never write "% of spec" and never call latency healthy or underperforming. It goes under **Findings → Needs operator input**: a link to the latency table, not its numbers again, and what to tell the agent — the provider's expected latency for this fabric at this node count, or the path to a `torch-dist-bench.py --collectives all` console from a known-good cluster of the same fabric and node count, which the agent then compares collective by collective.

## Storage

### Introduction

Open Storage with **`### Existing Partitions` first**, then **`### Introduction`**. They are sibling headings, not nested. Never open with a fio table.

- **`### Existing Partitions`:** inventory only, as a **table** — `| mount | size | type | device / server |`. Deduplicate by **backing volume**, not by `df` line. Same RAID/device, same Lustre `nid@net:/fsname`, same CSI/PVC, or a bind-mount of another path = **one row**. Listing both `/` and `/tmp` (or two bind points of one share) looks like twice the disk. **Always include `/`** as the on-node-disk row, even when the local `fio-scan` ran under `/tmp` — `/tmp` is a scan path, not a second volume. **`type` is the backing filesystem** (`xfs`, `ext4`, `lustre`, …), not the container layer: `df -T /` often says `overlay` inside a pod — look through with `findmnt` / `lsblk -f` / the same-device `/tmp` mount and put that FS in the type column. Overlay is not a partition type. Order: `/` first, then one row per distinct shared volume, using the mount the cluster actually uses for work. Never loose paragraphs, never an “Also mounted” bucket. No `fio-scan`, no “this is the ceiling”, no “we poked this path”; nothing about what the eval did. Facts that apply to several rows (all one product, separate servers, a protocol shim to look behind) go in a sentence **after** the table. Do not bring in filesystem names from other clusters or past jobs.
- **`### Introduction`:** same convention as Compute and Network: a **numbered list of passes that ran**, each named by tool + path, not a list of mounts. GitHub-link [`fio-scan`](https://github.com/stas00/ml-engineering/blob/master/storage/fio-scan) on the first item and say what it does there (six `fio` runs over 16 KiB / 1 MiB / 1 GiB × read/write; 16 jobs, 4 KiB blocks, `libaio`, `O_DIRECT`; 3 min each). Later `fio-scan` mentions are bare `code`. Typical items: local `fio-scan` on `/tmp` (the baseline for the shared table's **local/shared ×** columns), shared `fio-scan` on the work FS, and — if **≥2 nodes** — a concurrent 1 GiB write on each share `fio-scan` measured, **one writer per given node** (tool is `dd` unless you used something else). Omit the concurrent poke on 1 node. Run the same `dd` from node0 alone first, seconds before the concurrent one: `dd` is one stream with one request in flight, so `fio-scan`'s rates can't serve as its baseline, and without its own single-node figure the report can say how evenly the share served the writers but not whether they slowed each other.

**Report what you did, not what you didn't.** A mount you list for inventory needs its path, type and size — not "not fio-scanned", "not benchmarked", "not tested". The reader assumes anything without a measurement wasn't measured; spelling it out adds a negative to every line and buries the results. Same for tests that were never part of the plan. **Never assert an absence** — no `Skip reasons: none`, no `### Gaps` / `None`, no `N/A` rows, no "nothing was skipped". If it isn't there, it isn't there; the reader can see that. When a required phase genuinely could not run, the reason goes in **### Gaps** and nowhere else.

**Never report how full a partition is** — no `Use%`, no "X% full", no "N T free", nowhere in the report (not the Storage table, not the intro paragraphs, not Findings). It is a snapshot of what other tenants happened to be storing that hour, it has nothing to do with whether the hardware performs, and it will be wrong by the time anyone reads it. From `df` take only **size, type, path, and server/device**.

Shared network filesystems are many different products. Report **the one this allocation uses**. `df -T` is the guest-visible type (protocol or native FS); that may or may not be the vendor product. If the type is a share protocol rather than the product (common with virtiofs: [project](https://virtio-fs.gitlab.io/), [kernel](https://docs.kernel.org/filesystems/virtiofs.html)), recover the product from mount tag (`findmnt` SOURCE, `/sys/fs/virtiofs/*/tag` when it is virtiofs), pod volumes (`hostPath` vs PVC), node CSI annotations, and vendor docs — then link those docs. Advertised bandwidth may sit in the intro as product inventory. **Measured vs advertised** belongs in **Findings → Underperforming subsystems**, not here — or in **Findings → Needs operator input** when no advertised figure is known.

**Decide which shared mounts to scan by grouping them, and scope every claim to the mount it came from.** Managed parallel filesystems are usually provisioned per TiB of capacity, so a 2 TiB share and a 25 TiB share are not the same product at a different size — they have different expected rates, and a result from one does not transfer to the other. Group the rows of your inventory by **(backing type, size)**:

- Same type and same size → assume they perform alike. Scan **one**, say in the intro that the others are the same type and size so the numbers stand for the group.
- Different size or different type → a separate `fio-scan` each, or the group is unmeasured. At ~19 min per mount this is the cost of a real answer; ten distinct mounts is a long pass, not an excuse to generalize from one.
- Short on time? Scan the one the workload will actually hammer (checkpoints go to the big share, not the code share) and say plainly that the figures describe **that** mount, naming its size. Then the ask in **Needs operator input** covers that mount alone.

After scanning, group tables by **measured equivalence**, not by impression: two mounts are equivalent only if **every corresponding bandwidth and IOPS cell** (16 KiB / 1 MiB / 1 GiB × read/write) differs by at most **5%**. Use `abs(a-b) / min(a,b) × 100` so the test is symmetric. Equivalent mounts share one table whose heading lists every mount and size; mounts that fail even one cell get separate tables. State the rule and whether any mounts grouped. Never average across mounts, never present one mount's table as "the shared FS", and never ask the operator for expected figures for mounts you did not measure.

**On a parallel FS, capture the layout while you are on the node — it is the difference between a number and an explanation.** A single-client `fio-scan` result is capped by how many storage targets the file is spread over, so collect it in the same pass as the scan and keep it in `raw/`: on Lustre `lfs df -h <mount>` (how many OSTs and their sizes) and `lfs getstripe -d <scan dir>` (default stripe count and size); the equivalents are `mmlsdisk` / `mmlsfs` on GPFS, `beegfs-ctl --getentryinfo` on BeeGFS, `nfsstat -m` for NFS mount options. Stripe count 1 with a fast fabric and a slow result is the finding, not a mystery. Without it you can only ask the user to explain their own numbers.

Published recipe is 3 min runtime (`fio-scan` default, ~18 min/mount). First-pass trial may pass `FIO_RUNTIME=60` only if you edit a **copy** of `fio-scan` — never silently change the published script. Run twice: NVMe path, then shared path. `--unlink=1` is already in the script; do not reuse leftover fio files.

**Every hand-rolled `fio` command needs `--unlink=1`, and every scan leaves scratch behind if it doesn't.** `--numjobs=16 --filesize=1g` writes **16 GiB per run**, so one six-run scan is ~33 GiB **per mount** — five mounts is 165 GiB of other people's storage, on filesystems that are often nearly full. `fio-scan` passes `--unlink=1` for exactly this reason; the moment you inline your own `fio` loop (to sweep several mounts, to add a size) that flag is the first thing to carry over. Verify with `du -sh <scan dir>` after the first run rather than at the end of a 90-minute sweep.

```bash
./fio-scan "$NVME_PATH"
./fio-scan "$SHARED_PATH"
```

Need `python` on PATH for `fio-json-extract.py`, or patch the copy to `python3`.

**Always format the report tables the same way** for the shared core columns (local NVMe and every shared FS group). They must be comparable at a glance:

- Column names: **`latency (msec)`**, **`bw (MiBps)`**, **`IOPS (M)`**. Units live in parentheses in the header, not the cells. Never `lat msec` — write **latency**, and put **msec** in `()` like the other unit columns.
- **bw** stays MiBps in both tables. Never write `k`/`M` on a MiBps value. Fast/local table: `_` thousands (`12_029.1`). Shared/slow: ungrouped is fine (`631.8`).
- **IOPS** is millions (10⁶) in **both** tables (`3.08` vs `0.157`). Do not mix `k` in one table and `M` in the other. Do not put `M` in the cells.
- Shared tables only: append **`local/shared ×` at the end** (local bandwidth ÷ shared bandwidth, one decimal) — **how many times slower**, not a percentage. A fraction-of-local percentage (12.4%) makes the reader do the reciprocal in their head; 8.0× is the number they want. Never `%local`. Core columns stay the same order as local: `size | rw | latency (msec) | bw (MiBps) | IOPS (M)` then the `×` column. 1 = as fast as the local table’s same row. Do not put it on the local table. With a fixed block size the IOPS ratio is identical, so a second IOPS-ratio column repeats the same information.
- Full-precision unscaled numbers stay in `raw/` (`fio-nvme-summary.md`, `fio-shared-summary.md`).

## Report template

**Assume the reader knows nothing you learned during the eval.** Before any number, say what the tool literally did — what it ran, how many times, what it varied. Never let a term appear that the report has not defined, and **expand every acronym at its first use** (MAMF, MSMF, DCGM, NVLS, GEMM, busbw, algbw): "any shape" is meaningless until the reader has been told the tool searches **matmul shapes** (M×N×K); the same goes for `O_DIRECT`, CTA/wave, `local/shared ×`. Write the definition where it is first used, not in a glossary the reader has to hunt for. If a sentence only parses because you watched the run, rewrite it.

**Cut phrases that carry no information.** "and collects the results into one table", "the results are shown below", "for reference", "as expected", "it is worth noting", "this section covers…" — the reader can see the table; saying it exists adds nothing. Every sentence must add a fact, a definition, or a consequence. If deleting it loses nothing, delete it.

**Detail what is abnormal; keep what is normal to one line.** The report has to read the same on 4 GPUs and on 1000, so never give a row to each GPU or each node whose result is normal. For a per-GPU or per-node measurement write the median and the range, with where the lowest and highest are, and give rows only to the outliers: what was flagged, failed, crossed a candidate bar, or is the lowest of a pass a later pass flags. An abnormal item gets everything: its numbers in every run, the bar it crossed, its identifiers. A healthy subsystem gets one sentence. The full per-GPU and per-payload data stays in `raw/`, and the section links it. More words and numbers are not more information: a 32-row table of normal GPUs hides the one that isn't.

`reports/<cluster>-<YYYY-MM-DD-HHMM>.md` plus `reports/raw-<name>/` (consoles, txt, fio json, plots), next to this skill or in the working directory. Example: `h200-2026-09-17-1238.md`.

**Times are in the time zone of the machine the agent runs on**: `TZ` if it is set, otherwise that machine's system zone (`date +%Z`), not the nodes', which are often UTC. The filename's start time and the `Datetime` line use it, and the line names it, e.g. `2026-09-17, 12:38 PM to 2:51 PM CST`. Logs print node time, so convert their timestamps before quoting them.

**`### Gaps` under Compute, Network, or Storage:** omit the subsection entirely when that phase is complete. Never write `### Gaps` / `None`. Only include it when something is actually missing or incomparable (busy siblings, lemon GPU, DCGM `-r 2` failed/unprivileged, rdzv failed on ≥2 nodes, fio missing, …). **1 node is not a gap** — inter-node is omitted, not missing. When the rest of the report is written, **ask the user to address every remaining Gaps item**, each with a **proposed plan of action** (loop step 9).

````markdown
# Cluster eval: <name> (<N> nodes of <G>× <GPU>)

Example titles: `4 nodes of 8× H200`, `1 node of 8× B200`, `16 nodes of 8× H100`. Always include **node count and GPUs per node**, not only the GPU name.

- Datetime (start to end, in the zone above, named):
- Access:
- Nodes / GPUs:

Then a one-line pointer to the raw logs directory: `Raw logs: [\`raw-<name>/\`](raw-<name>/).`

## Table of Contents
Numbered list **before Environment**. Each item is **topic — tool/test that ran**, linking to that heading (GitHub-style `#anchor`: lowercase, punctuation stripped, spaces to `-`). Duplicate titles (`### Introduction`) get `-1`, `-2` suffixes; prefer linking the unique bench heading, not the intro. Only list tests that actually ran (plus inventory).

Compute TOC links are **purpose**, tools after the dash. Do **not** use DCGM / MAMF / Lemon GEMM as the TOC link text; those names live in the annotation or the section body. The `###` headings must match the TOC labels so the anchors agree.

The connectivity test has no section of its own — its result is item 1 of the Network `### Introduction`, so that TOC line links to `#introduction-1` (the second `### Introduction` in the document). Label it by the ranks it actually ran on: **Inter-node connectivity** only if it spanned every node, otherwise **Intra-node connectivity** with the rank count.

Network TOC: **All-reduce benchmark** and **Latency benchmark** — `torch-dist-bench.py`, then the rank counts (and "every collective" for the latency one). Do not drop the word **benchmark**. On 1 node only 8 ranks.

Storage TOC: **Existing Partitions** — mount inventory. Then **Local disk IO benchmark** — `fio-scan` on `<local path>` and **Shared <product> IO benchmark** — `fio-scan` on `<shared path>[, <shared path>…]` (same shape; paths after `on`, never inside the link). Concurrent poke when ≥2 nodes: **Concurrent write shared fs benchmark** — 1 GiB `O_DIRECT` `dd` from node0 alone, then from all `<N>` nodes, on each share.

Example:

```markdown
## Table of Contents

1. [Environment](#environment) — host, GPU, fabric, mounts, software (inventory, not a bench)
2. [Compute](#compute)
   1. [Hardware health](#hardware-health) — `dcgmi diag -r 2`
   2. [GPU performance benchmarks](#gpu-performance-benchmarks) — `mamf-finder-all-gpus.py`, one GPU at a time while the others run a matmul
   3. [Find underperforming GPUs](#find-underperforming-gpus) — fixed 16384³ `matmul` on every GPU, then `gpu-straggler-finder.py` on all 32 at once
3. [Network](#network)
   1. [Inter-node connectivity](#…) — `torch-distributed-gpu-test.py`, R ranks (label it **Intra-node connectivity** if it only ran on one node)
   2. [All-reduce benchmark](#all-reduce-benchmark) — `torch-dist-bench.py`, 8 ranks and 32 ranks (only 8 on 1 node)
   3. [Latency benchmark](#latency-benchmark) — `torch-dist-bench.py`, 8 ranks and 32 ranks, every collective
4. [Storage](#storage)
   1. [Existing Partitions](#existing-partitions) — mount inventory
   2. [Local disk IO benchmark](#local-disk-io-benchmark) — `fio-scan` on `/tmp`
   3. [Shared Lustre IO benchmark](#shared-lustre-io-benchmark) — `fio-scan` on `/code`, `/data`
   4. [Concurrent write shared fs benchmark](#concurrent-write-shared-fs-benchmark) — 1 GiB `O_DIRECT` `dd` from node0 alone, then from all 4 nodes, on each share (omit if 1 node)
5. [Findings](#findings)
   1. [Healthy subsystems](#healthy-subsystems)
   2. [Underperforming subsystems](#underperforming-subsystems) (omit if nothing underperformed)
   3. [Needs operator input](#needs-operator-input) (omit if every target was known)
```

Rebuild the anchors from the headings you actually wrote. Omit TOC lines for phases that did not run — no placeholder line saying they are missing.

## Environment
Five small tables — **Host / GPU / Fabric / Filesystems / Software**. Do not pile every field into one table. The third and fourth are **Fabric** and **Filesystems**, never “Network” / “Storage”: those would take the `#network` / `#storage` anchors the top-level sections need.

### Host
**First row, on a cloud: the instance type**, since the provider's specs (local storage, NICs) are published per instance type: on Kubernetes the node label `node.kubernetes.io/instance-type` (`kubectl get node <node> -o jsonpath='{.metadata.labels}'`), otherwise the metadata service (`curl -s 169.254.169.254/latest/meta-data/instance-type` on AWS). **Then Ubuntu** (`PRETTY_NAME`, e.g. 24.04.2 LTS). Then hostname, kernel (`uname`), CPU cores/NUMA, host RAM. Not Python, not PCIe (it is the GPU ↔ NIC leg of the fabric, see **Fabric**).

### GPU
name × count, SM arch / SM count, HBM in **GiB only** (not `275040 MiB (268 GiB)`), driver / CUDA, **three** SM clocks — boost (`clocks.max.sm`), saturated (~TDP under dense GEMM), parked idle — power limit / idle draw. Do not merge boost and “loaded”; a dense GEMM is not at boost. No MAMF/MSMF names here. PCIe does **not** go here.

### Fabric
intra-node fabric + spec uni GBps; inter-node adapter + rate as **inventory** (`ibstat` / `rdma link`); and a **GPU ↔ NIC (PCIe)** row, the leg every inter-node byte crosses: how many GPUs and NICs share each PCIe switch, so how many NICs each GPU has on its own switch, the gen × width of the GPU and NIC links, and the NUMA node of each group. Read it from sysfs, not from `nvidia-smi topo -m`, which lists only the NICs of drivers it knows: on AWS P6 it showed the 2 InfiniBand ports and none of the 8 EFA NICs, so every GPU looked `SYS`-far from a NIC. For each GPU (`nvidia-smi --query-gpu=index,pci.bus_id,pcie.link.gen.current,pcie.link.width.current`) and each `/sys/class/infiniband/*`, `readlink -f` the PCI device: the paths that share a parent bridge are on one switch; `current_link_speed`, `current_link_width` and `numa_node` sit next to it. Save the list in `raw/pcie-map.txt`. A NIC with fewer lanes than its port rate needs (100 Gb/s on gen 4 ×2) or GPUs with no NIC on their switch go in **Gaps** if the job's traffic uses them.

### Filesystems
Left column **Local** / **Shared network FS** — do not repeat “local” or “ceiling” in the value. Local value: mount + size; if the backing device is Linux md, write `md127 RAID` (or whatever `mdN`), not a bare `md127`. Shared: guest-visible type + path + size; add the vendor product name when it is not the same as `df -T`. **Capacity only — never how much is used or free.**

### Software
Python version, torch, NCCL (`torch.cuda.nccl.version()`). Note that torch was the newest release for the driver's CUDA at eval time, and name the newer release if one exists for a newer CUDA only. **No eval venv path.** Isolated venv is how you run, not a report field.


## Compute

### Introduction
Name × count, SM count, HBM GiB, TDP, official BF16 TFLOPS. Not a benchmark dump. Then a **numbered list of the passes that follow**, each saying what the tool literally does:

1. `dcgmi diag -r 2` — NVIDIA's own health check (software, memory, PCIe).
2. `mamf-finder-all-gpus.py` — times many **matmul shapes** (M×N×K) on GPU0 to find the fastest, then measures the sustainable winner on every other GPU, each while all the others run a matmul; reports **MAMF** and **MSMF**, both vs the official spec.
3. Lemon GEMM — a throwaway inline script: one fixed matmul shape (M=N=K=16384) on every GPU, to catch a dead, slow, or throttled card.
4. `gpu-straggler-finder.py` — runs a Llama-3.1-8B training step (its 32 decoder layers, forward and backward, on one 8192-token sequence), a matmul and 2 memory-bound workloads on every GPU of every node at once, and compares each GPU with its node's median GPU in the same 2-second windows, and each node with the median node, to find a GPU that works but runs slower than its peers.

### Hardware health
**Pass** (or **Fail** on named plugins/GPUs; or Gaps: no hostengine / unprivileged). Never **Pass** and **No Fail** together.

### GPU performance benchmarks
Official TFLOPS source = [TFLOPS comparison table](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/README.md#tflops-comparison-table), not a relative path. Open that section by explaining the measurement before any number: compute is measured by timing matrix multiplication; how fast a `matmul` runs depends on its **matmul shape** (the M×N×K dimensions), so the tool searches. State how many matmul shapes GPU0's search timed (the `Tried N shapes` line in `mamf/gpu0.txt`), and that every other GPU then measured GPU0's two winning shapes while all the rest of the node ran a continuous matmul. Then `Two numbers vs official BF16 <N> TFLOPS on <GPU>:` (name the GPU; do not leave the spec floating), then two bullets, each with the measured W/MHz the script printed:

- **MAMF** (Maximum Achievable Matmul FLOPS) — highest TFLOPS a short burst at boost reaches, with the board far under its power limit, measured on every GPU at GPU0's MAMF shape; the node figure is the median GPU. A ceiling, not a rate any sustained workload holds. One number, with its shape and % spec.
- **MSMF** (Maximum Sustainable Matmul FLOPS) — best that survives once the matmul shape is large enough to pin the board at the power limit and the clock settles, measured on every GPU while all the others compute. The node figure is the median GPU, so one weak or strong chip doesn't set it; synchronous work runs at the slowest GPU, which the summary also gives. Compare a sustained workload to the median, and a synchronous job's ceiling to the slowest.

Then one table over all the nodes, from the `mamf/summary.txt` of each, and a row for each GPU more than 5% below its node's median MSMF, the candidate bar (none: say so in the sentence after the table):

| | median | lowest | highest |
| --- | ---: | ---: | ---: |
| node MAMF (TFLOPS) | … | … (`<host>(node<I>)`) | … (`<host>(node<I>)`) |
| node MSMF (TFLOPS) | … | … | … |
| GPU MSMF vs its node's median | | -…% (`<host>(node<I>):gpu<N>`) | +…% (…) |

TFLOPS are integers, as the script prints them; compute % spec from the integer. The per-GPU tables stay in `raw/mamf-*/summary.txt`. On a 1-GPU box there is no "while the others compute" and no median or spread; say it is the only GPU.

**Expand every acronym at first use in the report** — MAMF, MSMF, DCGM, NVLS, GEMM, busbw/algbw, MFU. The Terms table in this skill is for the agent; the reader of the report only sees the report.

**The eval does not know what the cluster is for.** Never assume training — the same allocation may be bought for inference, fine-tuning, or batch jobs. Write "sustained workload", not "training step"; say "training or inference" when you need an example.

Give the winning shapes and the power and clock they ran at in the bullets, as ranges over the nodes.

### Find underperforming GPUs
**TLDR:** `no under-performing or dead GPUs` — name both outcomes, since a dead card and a slow one are different findings. If something is wrong, say which: `under-performing: <host>(node<I>):gpu<N>`, `dead: <host>(node<I>):gpu<N>` or `slow node: <host>(node<I>)`. Then define an under-performing GPU (works correctly but runs slower than its peers, which every GPU of a synchronous job then waits for; dead is the extreme case), and say in one sentence why the GPU performance pass can't settle it: MAMF and MSMF report a GPU's best stretch. Do **not** prefix the heading with node count — ranks belong in the TOC annotation, not the heading. Then two `####` subsections, in the order they ran:

- `#### Lemon GEMM` — the **same** matmul shape (M=N=K=16384) on every GPU, siblings idle, flag a card well below its node's median — explicitly **not** MAMF (no search over matmul shapes, no boost/sustain split). Then 16384³ SM coverage, then the one sentence of **Lemon GEMM** above (median, range, lowest GPU), and rows only for GPUs that tripped the gate.
- `#### Stragglers` — what `gpu-straggler-finder.py` literally ran before any number: the 4 workloads with their sizes and durations, all GPUs at once, the 2-second windows, each GPU against its node's median GPU in the same window and each node's median GPU against the median node's, and the flagging rule in words — define the **modified z-score** where it is first used (how far a GPU is from the median of all the GPUs, in units of their median absolute deviation, scaled so that 3.5 is a usual cut-off for an outlier) and the **90th percentile window** (the window that only a tenth of a GPU's windows are slower than). Then the nodes in one sentence unless one was flagged: the highest any node's median GPU reached against the median node's, and in which workload, against the 3% bar. Then a table of the **flagged GPUs only**, as the script printed their rows (`<host>(node<I>)`, GPU index, the 4 workloads' median / 90th percentile window against the node's median GPU, step and matmul clock, power, temperature, flag), and one last row, **slowest of the other N**, with the highest median / 90th percentile window any unflagged GPU reached in each workload and which GPU it was: that row is the healthy spread the flagged ones stand out from. Never every GPU's row; they stay in `raw/straggler.console`. Then the run count and the verdict per flagged GPU: the workload, its numbers in each run, the runs it was flagged in, its UUID and serial. A GPU flagged in some runs and not in all gets that one sentence here and nothing in Findings.

## Network

### Introduction
Open with a **numbered list of tools** (same shape as Compute and Storage), each explained before any number: GitHub-link [`torch-distributed-gpu-test.py`](https://github.com/stas00/ml-engineering/blob/master/debug/torch-distributed-gpu-test.py) (connectivity — one process per GPU; ranks see each other and complete a collective; say how many ranks and over how many nodes it ran, and do not call a 1-node run inter-node) and [`torch-dist-bench.py`](https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/torch-dist-bench.py) — say what it sweeps (all nine collectives, payloads from 8 B to 16 GiB) and define `busbw` (bus bandwidth, what the fabric moves), `algbw` (algorithm bandwidth, what the caller sees) and the latency median and p99 (the time of one call, the slowest rank's, over up to 1000 calls) **there**, plus which rank counts it ran at. Do not write “the book” or “MLE”. Then NVLS (see Terms): in-network **all-reduce** on the NVSwitch (~30% intra / ~25% inter vs ring `busbw`); does **not** help other collectives. InfiniBand switch SHARP is a different feature. Record `NCCL_NVLS_ENABLE`: **2** = auto — the eval default if unset; **0** = force off; **1** = require NVLS (can abort if unavailable). State whether NVLS was available for intra and for inter (`NCCL_DEBUG` when you have it). Name who set the env (container image / cluster), not that you “kept” it.

### All-reduce benchmark
One sentence per scope, then its plot. Intra-node: transport (e.g. NVLink 5, NV18), spec uni GBps, peak busbw and its payload, % spec (note NVLS if on). Inter-node **only if ≥2 nodes**: NIC inventory GBps, NCCL path (not % of NIC spec), peak busbw vs 1 node's at the same payload. No busbw table when the curve is normal: the plot shows it, and the console in `raw/` has every payload. A table only for the payloads that fell short or behaved oddly (a dip, a mid-size payload slower than a smaller one). Do not name the launcher. Rank/node counts go in the TOC annotation.
**Inline** `![Intra-node all-reduce](raw/<cluster>/busbw-mean-all_reduce-…-8.png)` and, with ≥2 nodes, `![Inter-node all-reduce](raw/<cluster>/busbw-mean-all_reduce-…-R.png)`.
### Latency benchmark
Why all-reduce alone would flatter, and what `batch_isend_irecv`, `all_to_all`, `all_reduce` and `all_gather`/`reduce_scatter` each show (see **Latency**), in as few sentences as that takes, then one table of all nine collectives at the smallest payload: the intra-node median, and with ≥2 nodes the inter-node median and `inter/intra ×`. p99 only in a sentence, for a collective whose p99 is more than 1.5× its median. The PyTorch caveat for `batch_isend_irecv` and `scatter` in one clause. No verdict here or in Findings beyond **Needs operator input**.
**Inline** `![Intra-node latency](raw/<cluster>/latency-collectives-…-8.png)` and, with ≥2 nodes, `![Inter-node latency](raw/<cluster>/latency-collectives-…-R.png)`.
### Gaps
(only if needed, e.g. rdzv failed on ≥2 nodes — not “only 1 node”)

## Storage

### Existing Partitions
Inventory table: **always a `/` row**, then one row per **distinct** volume. Same backing store mounted twice (`/tmp` on `/`, bind-mount aliases) is still one row. No benchmarking comments. Do not name a network FS this allocation does not use. Shared facts go after the table.

| mount | size | type | device / server |
| --- | ---: | --- | --- |
| `/` | 28T | xfs | md127 RAID0 (8× ~3.5T NVMe) |
| `/code` | 2.3T | Lustre | `10.0.0.100@tcp:/fsname` |

### Introduction
Same shape as Compute / Network: **numbered list of passes** (tool + path), not a mount inventory. GitHub-link `fio-scan` on the first item and explain what it literally does **there** — six `fio` runs against one directory — file sizes **16 KiB, 1 MiB and 1 GiB (correspondingly small, medium and large)**, each read and write — 16 parallel jobs, 4 KiB blocks, `libaio`, `O_DIRECT` (and say what `O_DIRECT` buys: bypasses the page cache so the numbers are the filesystem's, not RAM's), 3 min of timed IO each.

**Write IEC units, never `fio`'s shorthand.** The script prints `16k` / `1m` / `1g`; the report says **16 KiB** / **1 MiB** / **1 GiB**, in the `size` column of both tables and in every sentence that cites a row. `16k` is not a valid unit. Later mentions are bare. Then the table units note. All “we scanned X” comments live here.

Two tables, **same columns and same IOPS scale**:

| size | rw | latency (msec) | bw (MiBps) | IOPS (M) |
| ---: | --- | ---: | ---: | ---: |
| 16 KiB | read | … | 12_029.1 | 3.08 |
| … | … | … | 631.8 | 0.157 |

### Local disk IO benchmark
…
### Shared <product> IO benchmark
| size | rw | latency (msec) | bw (MiBps) | IOPS (M) | local/shared × |

Heading is **Shared <product> IO benchmark** (e.g. Shared Lustre IO benchmark), not a bare “Shared”. Several shares of one product go under that one heading, a table per share (or per group the 5% rule allows), each opened by a line naming it and its size (`` `/data`, 14T: ``); not a heading per share, whose duplicate names would break the anchors. Optional extra when ≥2 nodes: **Concurrent write shared fs benchmark**, one table for every share: `| share | node0 alone (MiBps) | 4 nodes at once, slowest – fastest node (MiBps) | 4 nodes together (MiBps) |`. Omit on 1 node.

Local NVMe vs shared network FS (16 KiB / 1 MiB / 1 GiB, read+write). bw = MiBps (`_` on the fast table). IOPS = 10⁶ in every table. **`size` is right-aligned** (`---:`), like every other numeric column; only `rw` is left-aligned. Each shared table appends **`local/shared ×`** at the **end** (local bandwidth ÷ shared bandwidth) so the first five columns match local. Explain the column in a **sentence**, not a chain of equations: what it divides and what 1 would mean. Do not add an IOPS ratio: fixed 4 KiB blocks make it identical to the bandwidth ratio. **Quote table values verbatim** in prose — if the column says `26.5`, write `26.5×`, never `27×`. Do not express the comparison as a percentage, do not call it `%local`, and say "the local disk table above", not a device name. Raw unscaled numbers in `raw/`.
(omit Gaps unless something is missing)

## Findings

Up to three headings, in this order — **### Healthy subsystems**, **### Underperforming subsystems**, **### Needs operator input** (the noun is required; never a bare `### Healthy`). Include only the ones that have bullets. Bullet labels are the subsystem (GPUs, fabric, local disk, shared network FS, torchrun) — **never prefix with A./B./C.** (the report sections are Compute, Network, Storage). Split mixed areas (e.g. local disk vs shared FS). Do not put an underperforming item in Healthy with a caveat, and do not hide it in a Flags list that looks like the healthy bullets.

Healthy subsystems: fabric and GPUs that met spec. Network headline is **fabric** (intra vs NVLink spec + NVLS; inter busbw + path + vs 1-node all-reduce) — **never** “X% of NIC/rail,” and **not** a launcher. Do not credit DeepSpeed / torchrun / srun in busbw or Findings.

**Under-performing and dead GPUs** get one bullet each under Underperforming subsystems, from any of the 3 passes (the lemon gate, the GPU performance pass, the finder). Name each as `<host>(node<I>):gpu<N>`, e.g. `blah-176(node0):gpu7`: `<host>` is the hostname the Environment Host table uses, which an operator needs to drain or replace the card; `node<I>` is the node's 0-based position in the hostfile (the order the user gave the nodes, from `raw/hostfile-map.txt`; `node0` on a 1-node eval, which has no hostfile), which stays readable when hostnames are cryptic; `<N>` is the GPU index `nvidia-smi` shows. Follow it with the pass that found it and its numbers against its node's median, e.g. "`blah-176(node0):gpu7` — MSMF 701 / 698 / 704 TFLOPS in three passes, 6.8–7.2% below the node median each time". Add every such GPU's UUID and serial, which `gpu-straggler-finder.py` prints on its line, and which the environment dump's `uuid` and `serial` columns give for a GPU another pass found: the serial is what an operator replaces the card by, and the UUID stays the same when the node is renamed or re-imaged, while the hostname and index may not — e.g. "`blah-176(node0):gpu7` — training step 3.4% / 4.1% / 3.2% / 3.8% slower than its node's median GPU in 4 runs, UUID `GPU-…`, serial `…`". A slow node gets its own bullet, `<host>(node<I>)`, with its median GPU against the median node's in each run and its GPUs' serials, which the finder prints. List every one; never summarize as "two slow GPUs".

Underperforming subsystems: measured vs **vendor advertised** for **this** product; not good enough for code (16 KiB) / checkpoint (1 GiB → 1 TiB) / dataloader; torchrun/rdzv if it failed; leftover Gaps. A broken launcher is **not** “the network is unhealthy.” Do not name a network FS this allocation does not use.

**Needs operator input: a verdict requires a target, so a measurement without one goes here, not under Underperforming.** Calling a subsystem slow when you have nothing to compare it against is an opinion dressed as a finding — the reader cannot act on it and the operator can dismiss it. This heading is for a subsystem that was measured cleanly but whose expected figures only the operator has: a shared FS with no published provisioned throughput, a fabric whose purchased tier is unknown, any device whose spec sheet does not exist publicly. Open with one sentence saying these cannot be called healthy or underperforming until those figures arrive.

**Every bullet ends with what the operator tells the agent, and what the agent then does with it.** A list of what the eval couldn't settle, with hints, leaves the operator nothing to do; the bullet is an instruction. Each one has 3 parts:

1. The subsystem and its measurement in one line, or a link to the section's table when it has more numbers than fit on one.
2. **Tell the agent:** a reply the operator can fill in and paste, in the units and under the conditions measured, with the blanks marked — e.g. `/code expected, one client, 16 jobs, 4 KiB blocks, O_DIRECT, MiB/s read / write: 16 KiB _ / _, 1 MiB _ / _, 1 GiB _ / _; per client or per filesystem: _`. Offer the alternatives the agent can work from, cheapest first: a product and tier the agent can turn into figures (`FSx for Lustre, 250 MB/s per TiB`), the figures themselves, or a path to a reference run of the same tool on a known-good cluster of the same type.
3. **The agent then:** what it does with the answer — compares each measured cell with its expected one, calls those within 10% healthy and the rest underperforming, and moves the bullet to **Healthy subsystems** or **Underperforming subsystems** with the comparison; or reruns what the answer unblocks (`lfs getstripe` once `lfs` may be installed).

Don't put here what the agent can do itself: install a missing tool if it has `sudo`, look up a product's published tier figures, rerun a pass. Do it, or put it in **Gaps** with the command it needs permission to run.

**Write targets in the plural — a filesystem never has one.** Every file size and operation has its own rate, reads and writes differ, so it is "the figures they are supposed to hit" and "no published performance figures", never the singular. The singular quietly promises the reader one number to compare against, which is the mistake the whole category exists to avoid.

**Ask only about the mount you measured, and name its size.** Expected rates are per filesystem, so asking for "`/code` and `/data`" when six numbers came from `/code` alone invites an answer you cannot use. State which mount the figures describe, and if sibling mounts differ in size, say that each has its own target and needs its own scan — pointing at the one the workload will lean on hardest.

**Ask for the targets in the same shape as your measurement, or they will not settle anything.** "The provisioned throughput" is not askable — a parallel FS has no single rate, and a per-TiB headline number cannot be checked against a row of the table. Ask for **one expected value per measured cell, in the units you measured, under the conditions you measured**: for `fio-scan` that is bandwidth in MiB/s **or** IOPS at 16 KiB, 1 MiB and 1 GiB, read and write, from one client with 16 jobs, 4 KiB blocks and `O_DIRECT`. Either unit is enough because the fixed block size makes one derivable from the other. Say those conditions in the ask, because the same filesystem answers differently per client and in aggregate, and because small-file and large-file rates come from different limits — one headline figure hides both. Then add the conditions that would invalidate a comparison if the operator's figures assume something else: per-client vs per-filesystem, and stripe count. A comparison against another subsystem you did measure (shared FS vs local disk, `26.5×` slower) is a fact and belongs in the bullet; it is not a substitute for the target, because local NVMe was never what the shared FS was sold to match.

Judge before you place a bullet: is there a number this is supposed to hit? Official TFLOPS, NVLink spec GBps and a vendor-advertised FS figure are targets, so those bullets can be Healthy or Underperforming. If the answer is "the operator would have to tell me", the bullet goes in **Needs operator input** even when the measurement looks disappointing.

**Quote the whole measurement, compactly, not one cherry-picked row — or link the table.** A storage bullet has six numbers behind it, so give all six in one line — `read / write: 16 KiB **a MiB/s / b MiB/s**, 1 MiB **c MiB/s / d MiB/s**, 1 GiB **e MiB/s / f MiB/s**` — instead of a single size, which leaves the reader asking what the others were. Same for any bench that swept a parameter. Compact, not partial.

**Every number outside a table carries its unit.** Putting `MiB/s` once in the lead-in and then listing bare `1_646.1 / 1_658.3` is a table habit; in prose each value needs its abbreviation — `1_646.1 MiB/s / 1_658.3 MiB/s`. `read / write` once is fine. Same for GBps, TFLOPS, msec, ×. The table header is the one place a unit may appear only once.

**When vendor figures are missing, say the targets are unknown, say why none are published, and ask for them — but do not narrate the search.** Three things, because a measurement with no target cannot be called a shortfall, and a reader who knows the filesystem's name will reasonably assume a spec sheet exists: (1) state plainly that the provisioned figures are unknown and the measured numbers therefore have nothing to be judged against; (2) explain in a sentence why — for a parallel FS, **the software publishes no performance figures**; Lustre, GPFS/Storage Scale, BeeGFS and NFS are filesystem software, and throughput is a property of the deployment (count of object storage targets / servers, their backing devices, the network to them, per-file stripe count), so the number that exists is the **service tier the operator purchased**, typically sold per TiB of capacity (managed Lustre commonly 125–1000 MB/s per TiB); (3) ask the user for that tier **and** the default stripe layout, since a single-stripe file is served by one storage target regardless of filesystem size and is the usual explanation for a single client seeing a few hundred MiB/s. What to leave out is the search: where you looked, what `df` or the mount table did not contain. A mount was never going to state IO speed, so ruling it out tells the reader nothing and reads as an excuse.

**Say who is slowed and by how much, in literal terms.** Hardware and files do not "feel", "suffer", "struggle", "get punished", or "care"; a filesystem is not "painful". Name the concrete operation and attach the measured rate to it — "a `git clone` or `pip install` on `/code` runs at the 16 KiB read rate of 204.7 MiB/s" — so the reader can check the claim against the table instead of trusting an adjective.

If nothing underperformed, **omit that heading and its TOC line** — never write `Underperforming subsystems: none`. Same for **Needs operator input** when every target was known. No extra **Flags** section.
````

## Final pass before you show the report

Prose rules get skipped. **Run this list against the finished report, line by line, before you tell the user it is ready.** Each item has failed a real review.

| check                                                                                                          | fix                                                                                                                                                  |
| -------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Every acronym expanded at first use (MAMF, MSMF, DCGM, NVLS, GEMM, busbw, algbw)                               | `Maximum Achievable Matmul FLOPS`, …                                                                                                                 |
| Every tool says what it literally does before its first number                                                 | how many shapes / payloads / runs, and what it varied                                                                                                |
| No term used before it is defined ("any shape", `local/shared ×`, CTA, `O_DIRECT`)                             | define at first use, in a sentence — not as an equation                                                                                              |
| Shared-vs-local stated as **times slower**, not percent                                                        | `8.0×`, not `12.4%`                                                                                                                                  |
| Prose numbers identical to the table cell they cite                                                            | table `26.5` → prose `26.5×`, never `27×`                                                                                                            |
| Findings bullets carry the full sweep, compactly                                                               | all three sizes × read/write on one line, not just 1 GiB                                                                                             |
| Every number outside a table has its unit                                                                      | `1_646.1 MiB/s / 1_658.3 MiB/s`, not a lead-in `MiB/s:` then bare numbers                                                                            |
| Nothing inanimate given feelings                                                                               | not "small files will feel that" — name the operation and its measured rate                                                                          |
| Targets in the plural                                                                                          | "the figures they are supposed to hit", never "the figure"                                                                                           |
| Missing vendor figures: target-unknown admission + why none are published + the ask, no search narrative       | keep "provisioned throughput is unknown, so there is no target to compare against"; drop "the mounts show only the server address and size"          |
| Latency reported for every collective, never as % of spec or a verdict                                         | all nine at the smallest payload; `inter/intra ×` when ≥2 nodes; Findings → **Needs operator input** only                                            |
| Every **Needs operator input** bullet is an instruction                                                        | ends with **Tell the agent:** (a fill-in reply, in the measured units) and **The agent then:** (what it compares or reruns, and where the bullet moves) |
| No row for a normal GPU, node or payload                                                                       | median and range in a sentence; rows for outliers only, plus a "slowest of the others" row; the rest in `raw/`                                       |
| No bullet judged without a target                                                                              | no known figure to hit → **### Needs operator input**, not Underperforming                                                                           |
| Shared-FS figures scoped to the mount they came from, with its size                                            | one mount's table is never "the shared FS"; differently-sized siblings each need their own scan                                                      |
| Distinct shared mounts grouped only by the 5% rule                                                             | one table only if every corresponding bandwidth and IOPS cell is within 5%; otherwise separate tables                                                |
| The ask names one expected value per measured cell, with conditions                                            | "expected MiB/s or IOPS at 16 KiB / 1 MiB / 1 GiB, read and write, one client, 16 jobs, 4 KiB blocks, `O_DIRECT`" — not "the provisioned throughput" |
| Sizes in IEC units                                                                                             | `16 KiB` / `1 MiB` / `1 GiB`, never `16k` / `1m` / `1g`                                                                                              |
| Storage table headers carry units in `()`                                                                      | `size`, `rw`, `latency (msec)`, `bw (MiBps)`, `IOPS (M)`                                                                                             |
| Numeric columns right-aligned, including `size`                                                                | `---:` for all but `rw`                                                                                                                              |
| Every `###` heading matches its TOC label, and every anchor resolves                                           | rebuild anchors from the headings you wrote                                                                                                          |
| Labels match what actually ran                                                                                 | a 1-node connectivity run is **intra**-node                                                                                                          |
| Every slow or dead GPU named as `<host>(node<I>):gpu<N>` in Findings, and a slow one confirmed in three passes | `blah-176(node0):gpu7`, not "GPU7" or "one slow card"                                                                                                |
| A straggler in Findings only if `gpu-straggler-finder.py` flagged it in all 4 runs, with its UUID and serial   | flagged in some runs → one sentence in `#### Stragglers`, not a Findings bullet                                                                      |
| No absence asserted                                                                                            | no `Skip reasons: none`, `### Gaps` / `None`, `N/A`                                                                                                  |
| No redundant verdicts                                                                                          | `Pass`, not `Pass … No Fail`                                                                                                                         |
| No filler sentences                                                                                            | "collects the results into one table", "as expected"                                                                                                 |
| No workload assumed                                                                                            | "sustained workload", not "training step"                                                                                                            |
| Scratch cleared from every mount you wrote to                                                                  | `du -sh` each scan dir; `fio-test/` moved to that mount's `trash/`, paths and sizes named in your summary                                            |

Then fold any correction the user still makes back into this file (see **Keep this skill up to date**).

## Training wheels

These are **examples of past jobs**, not defaults and not required knowledge. Discover the next cluster from scratch (how to exec, namespace, launcher, NCCL env, mounts). Skip this section if none of it matches.

- One AWS EFA allocation used `kubectl` in namespace `my-namespace`, hostfile `/data/hostfile`, SSH port **2222**, Lustre on `/code` `/data` `/checkpoint`, and image env `NCCL_NET_PLUGIN=ofi`, `NCCL_TUNER_PLUGIN=ofi`, `NCCL_NVLS_ENABLE=1`. `ibstat` was empty; `rdma link` showed `rdmap*`.
- One other trial used virtiofs on `/mnt/data`, container `sshd` **2222**, and hostname → `127.0.1.1` (torchrun needed `is_host` + `local_addr`).

Lessons:

- `pip install matplotlib` in the eval venv on **shared FS** so **every** node can plot. `torch-dist-bench.py` writes the PNG in the **process cwd** (often `$HOME` under torchrun), not the eval directory — copy it immediately. If a launcher still uses a different interpreter without matplotlib, regenerate the PNG from the printed table on the laptop.
- Environment is for a reader who has not seen MAMF yet: no MAMF/MSMF, no eval venv path, HBM in GiB, PCIe in Fabric as the GPU ↔ NIC row, OS first Host row, Python on Software, Local cell must not repeat “local”/“ceiling”, md devices as `md127 RAID`.
- Network opens with **### Introduction** (NVLS + who set `NCCL_NVLS_ENABLE`), not a bare env-var line.
- Find underperforming GPUs: TLDR first (**no under-performing or dead GPUs**); median, range and the outliers, not every GPU's score; 16384³ SM coverage **from this SM count**.
- Install `infiniband-diags` (`ibstat`), `fio`, `iproute2`/`rdma-core` (`rdma link`), `pdsh`, and DCGM **before** the env dump. Use `sudo` when not root. Empty `ibstat` on EFA is OK if `rdma link` works. If `dcgmi diag -r 2` cannot run, Gaps — do not skip MAMF.
- Shared-FS `fio-scan` 1 GiB × 16 jobs × `O_DIRECT` **can** take hours (file create/unlink on some network FS). A quiet 40 min is not necessarily a hang. Copy logs after the local-disk phase.
- Retry pip/`kubectl exec` on connection reset; do not treat RST as a failed install until a second probe (`import matplotlib`, `command -v fio`, `command -v ibstat` or `rdma`).
- `set -e` + `grep` with no matches aborts scripts (busy-GPU check). Use `grep -c … || true`.
- A `pkill -f` that cleans up a pass between steps must not match the launcher's own command line. `pkill -f <eval dir>` sent over `pdsh -w ^hostfile` also matched every `pdsh` whose hostfile lived in that directory, the cleanup's own included, and killed the DCGM install and the lemon GEMM about 10 s in. Match the interpreter instead (`pkill -f '[e]val-dir/venv/bin/'`) and run each pass for longer than the cleanup's sleep before trusting it.
- `du` on Lustre reports files written seconds earlier at a fraction of their size (4 freshly written 1 GiB files read 127K), so check scratch with `ls -la` or `du --apparent-size` before saying how much was left behind.
- Installing torch into a venv on a Lustre share took about 8 minutes (many small files); start it first, before the env dump.
- One `O_DIRECT` `dd` stream got 249–261 MiBps on 3 Lustre shares whose `fio-scan` 1 GiB writes differed up to 4×. The `dd` pair says whether writers slow each other, not how fast a share is; never rank shares by it.
- On FSx for Lustre (8-character filesystem names over `@tcp`, sizes in 1.2 or 2.4 TiB steps), `lfs` is not in Ubuntu's apt sources; it needs AWS's FSx client repository. Adding a repository to the nodes is the operator's call: put it in Gaps and ask with a one-line reply under **Needs operator input**.
- `gpu-straggler-finder.py --json` writes 20–65 MB with `--repeat 3`. `gzip` it on the node first, and if `kubectl cp` drops the connection partway, `split -b 10M` it on the node, copy the parts one by one, `cat` them together and compare `sha256sum` with the node's.
- Superseded artifacts go to a local `trash/<YYYYMMDD>-<topic>/` directory, never `rm` — wait for the user if the skill environment forbids delete.
