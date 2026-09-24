---
name: evaluate-cluster
description: >-
  Evaluates an ML GPU cluster for a cloud trial or acceptance test: environment
  dump, isolated newest PyTorch, sequential matmul FLOPS (MAMF/MSMF) on every
  GPU, intra-node all-reduce, inter-node all-reduce on every node you were given
  (omit that section if there is only one node), fio on local disk and shared FS,
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

Refuse to invent access, GPU counts, filesystem paths, or peer IPs. **Preflight first**: list what you have, **ask for every missing item, and wait**. Do not start DCGM, MAMF, `fio-scan`, or `all_reduce_bench.py` until the checklist is complete and — when **≥2 nodes** — inter-node reachability has passed. Eval **all nodes the user gave** (1, 2, 4, 16, …). Inter-node all-reduce uses **every** one of those nodes (`NNODES × GPUs_per_node` ranks). **One node:** do not write an Inter-node section at all (no “Skipped”). If ≥2 nodes but inter-node still cannot run after the user has answered, omit the section and put the reason in **### Gaps** — never invent busbw.

## Terms

| term | meaning |
| --- | --- |
| **MAMF** | Maximum Achievable Matmul FLOPS: the highest any **matmul shape** (M×N×K) reaches on that GPU. The winning shapes are short bursts — the clock sits at boost and the board stays far below its power limit — so this is a ceiling, not a rate any sustained workload holds. |
| **MSMF** | Maximum Sustainable Matmul FLOPS: what is left once the matmul shape is big enough to pin the board at its power limit and the clock settles below boost. This is what back-to-back dense matmuls actually get. Both come from [`mamf-finder.py`](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/benchmarks/mamf-finder.py), which times on the order of 1500–2000 matmul shapes per GPU (the `auto` search adapts, so the count differs per card) and reports the W and MHz behind each winner. |
| **DCGM** | NVIDIA Data Center GPU Manager. `dcgmi diag -r 2` checks GPU memory, bandwidth, PCIe — not FLOPS. |
| **lemon GEMM** | A one-off inline `torch.matmul` (not a published script) on every GPU with the same large shape, to catch a dead, slow, or throttled card. |
| **all-reduce** | Collective: every rank ends up with the sum of all ranks' tensors. Used to stress NVLink and the network. |
| **busbw / algbw** | Printed by [`all_reduce_bench.py`](https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/all_reduce_bench.py). **busbw** = unidirectional bus bandwidth (what the fabric moves). **algbw** = algorithm bandwidth (what the caller sees). busbw is already unidirectional ([why](https://github.com/stas00/ml-engineering/blob/master/network/README.md#unidirectional-vs-bidirectional-duplex)). |
| **NVLS** | NVLink SHARP: the NVSwitch does the all-reduce in the switch. Typical gain vs ring busbw ~30% intra-node / ~25% inter-node. Does **not** speed up all-gather or reduce-scatter. Different from InfiniBand switch SHARP. |
| **NCCL** | NVIDIA Collective Communications Library (the GPU collective stack torch uses). |
| **rdzv** | torch.distributed **rendezvous**: a TCP store so ranks find `MASTER_ADDR:PORT` before NCCL starts. |
| **hostfile** | Text file, one peer address per line (`10.0.0.1` or `10.0.0.1 slots=8`). DeepSpeed `-H` and `pdsh -w` read it. |
| **pdsh** | Parallel SSH: run one command on many hosts. |
| **fio-scan** | Wrapper around `fio` ([script](https://github.com/stas00/ml-engineering/blob/master/storage/fio-scan)): six timed runs (16 KiB / 1 MiB / 1 GiB × read/write). Needs [`fio-json-extract.py`](https://github.com/stas00/ml-engineering/blob/master/storage/fio-json-extract.py) in the **same directory**. |
| **`local/shared ×`** | local-disk bandwidth ÷ same-row shared-FS bandwidth — how many times slower the shared filesystem is. One column is enough: with a fixed block size the IOPS ratio is identical. Never a percentage, never `%local`. |
| **SM** | GPU streaming multiprocessor. Count from `torch.cuda.get_device_properties(0).multi_processor_count`. |
| **HBM** | GPU high-bandwidth memory. Report `nvidia-smi` `memory.total` in **GiB**. |
| **TDP** | Thermal design power: the GPU’s advertised watt limit (`power.limit`). |
| **EFA** | AWS Elastic Fabric Adapter (RDMA NICs named `rdmap*`). `ibstat` is often **empty**; use `rdma link`. |
| **overlay** | Container union filesystem. `df -T /` may say `overlay`; the **backing** FS (xfs, ext4) is the partition type to report. |

## Scripts (fetch, do not invent)

```bash
RAW=https://raw.githubusercontent.com/stas00/ml-engineering/master
curl -fsSL "$RAW/compute/accelerator/benchmarks/mamf-finder.py" -o mamf-finder.py
curl -fsSL "$RAW/network/benchmarks/all_reduce_bench.py" -o all_reduce_bench.py
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

**Goal:** be able to run the rest of the eval without blocking on the user. Slow work (PyTorch wheels, DCGM `-r 2`, sequential MAMF, shared-FS `fio-scan`, GPU all-reduce) starts only after this gate.

### Ask now, not later

Collect and **write down** every row. Missing → message the user with the exact list, then **stop**.

| need | why | typical answer |
| --- | --- | --- |
| How to exec on a node | every command | `kubectl` (ask for context, namespace, pod/job), `ssh`, `srun`, or whatever CLI the user names |
| **All** node identities | lemon GEMM + inter-node | every pod/hostname, not only rank 0 |
| **Hostfile (or equivalent) with peer IPs** | inter-node SSH / rdzv / `pdsh` | path or pasted contents: one IP per line. Required when **≥2 nodes**. SLURM: `scontrol show hostnames` plus IPs. |
| SSH / launcher details | `pdsh` / DeepSpeed | port; passwordless SSH **between** nodes (not only laptop → rank 0) |
| GPUs per node | default 8 | `nvidia-smi -L` if unknown |
| Local disk path + shared-FS path | `fio-scan` | discover with `df -hT`; ask if unclear |
| Advertised uni bandwidth / official TFLOPS | optional | else [accelerator tables](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/README.md) + `nvidia-smi` |

NCCL env is **not** a user question if you can exec: dump `env | grep -E '^NCCL_|^FI_|^AWS_OFI'` on rank 0 in this same preflight. Keep whatever the platform already sets for the fast path (`NCCL_IB_HCA`, `NCCL_NET_PLUGIN=ofi`, `NCCL_TUNER_PLUGIN=ofi`, EFA/OFI tunables, `NCCL_NVLS_ENABLE`, …). Do **not** overwrite a live platform value with the eval default. If NCCL is unset, **preset `NCCL_NVLS_ENABLE=2`**. Record the **final** env in the report (Environment + Network Introduction).

### Inter-node reachability (now, not a benchmark)

As soon as you have the hostfile/IPs, prove the nodes can see each other. **Do not** run `all_reduce_bench.py` or any GPU collective yet. This is seconds of TCP/SSH.

```bash
# 1. Launch node → every peer (BatchMode: fail fast if keys/port are wrong)
while read -r ip rest; do
  [[ "$ip" =~ ^# ]] || [[ -z "$ip" ]] && continue
  ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new "$ip" hostname
done < "$HOSTFILE"

# 2. Same via pdsh if that is the launcher
HOSTS=$(grep -v '^#' "$HOSTFILE" | grep -v '^$' | cut -d' ' -f1 | tr '\n' ',' | sed 's/,*$//g')
PDSH_RCMD_TYPE=ssh pdsh -w "$HOSTS" hostname

# 3. Peer → peer TCP (rdzv will need this). From node 1, hit node 0's IP:22 or the container sshd port.
MASTER_ADDR=$(grep -v '^#' "$HOSTFILE" | grep -v '^$' | head -1 | cut -d' ' -f1)
timeout 5 bash -c "echo >/dev/tcp/$MASTER_ADDR/${SSH_PORT:-22}" && echo OPEN
```

On Kubernetes, `ssh` may be `kubectl exec` per pod — still loop **every** sibling, not only the first (`-0`). Probe `sshd` listen port (`ss -ltnp | grep sshd`) and `ssh -p <port> <ip> hostname`. Do **not** assume port 22 or 2222. If PATH has a cluster SSH wrapper, use it; otherwise `ssh` / `pdsh`.

**Pass:** every hostname comes back, TCP to `MASTER_ADDR` is OPEN. **Fail:** ask the user (hostfile IPs, SSH port, keys, `pdsh` args) and wait. Do not start MAMF “in the meantime.”

After the eval venv exists (loop step 2), add one more cheap check **before** GPU inter-node all-reduce: CPU-only, 1-process-per-node PyTorch `gloo` backend, `init_process_group` + `all_reduce` (see Network). Still not `all_reduce_bench.py`.

## Loop

0. **Preflight.** Fill the checklist. Ask for what’s missing (hostfile with peer IPs is required for **≥2 nodes**; 1 node needs no hostfile). **Stop until the user answers.** Then, if ≥2 nodes, run the SSH/TCP reachability probes. Do not start the isolated install or any long bench until that passes (1-node evals skip the probes and skip inter-node entirely).
1. Dump environment (below) into the report draft.
2. Isolated venv on the **newest** released torch (do **not** mutate or benchmark the node’s preinstalled Python).
3. If ≥2 nodes: CPU gloo 1-process-per-node across the hostfile (connectivity + rdzv). Fix `is_host` / `local_addr` here if needed. **Skip this step on 1 node.**
4. Connectivity: `torch-distributed-gpu-test.py` — one process per GPU; ranks must see each other and complete a collective. **The scope is whatever you launch it on**: across all nodes it is an inter-node check, on a single node it is only intra-node. With **≥2 nodes, run it across all of them** — a 1-node run does not test the fabric and must never be labelled inter-node.
5. **Compute:** `dcgmi diag -r 2` (hardware health), then `mamf-finder.py --search auto` sequentially on each GPU. Lemon GEMM on **every GPU of every given node**.
6. **Network:** intra-node `all_reduce_bench.py`. Then, if **≥2 nodes**, GPU all-reduce on **all** of them (not a fixed 4). If **1 node**, stop after intra-node — no Inter-node heading.
7. **Storage:** `fio-scan` on local disk and on shared FS. Concurrent write poke uses every given node when ≥2; skip that poke on 1 node.
8. Write `reports/<cluster>-<YYYY-MM-DD-HHMMZ>.md` next to this skill (or in the working directory), with `reports/raw-<name>/` beside it and **plots inlined**. Write it for a reader who knows nothing you learned on the node: every tool and term gets defined where it is first used. End with **## Findings**, split into **### Healthy subsystems**, **### Underperforming subsystems**, and **### Needs operator input** — the last for subsystems measured cleanly but with no known target to judge them against; keep only the headings that have bullets. Compute may use % of official TFLOPS. Intra-node all-reduce may use % of NVLink spec (note NVLS). Storage: measured vs vendor advertised. **Never** flag multi-node all-reduce as “X% of NIC / rail spec.” Do not repeat the tables.
9. **Clear the scratch you created** (see **Leave the filesystems as you found them**) — before the summary, not after the user notices.
10. **Close-out:** **ask the user** to address each one, including every **Needs operator input** bullet — those are open questions addressed to them, and the eval stays unfinished until they supply the figures or say to drop it. For every gap: quote it, then give a **proposed plan of action** (commands, launcher, package, node count, or “accept and leave as-is”). Do not treat the eval as finished until they pick a plan, you execute it, or they explicitly accept the gap. If there are no Gaps sections, say the eval is complete.

Keep siblings idle during sequential MAMF. Abort a GPU's run if `nvidia-smi` shows other compute processes on that node (except the connectivity test / the bench itself).

## Leave the filesystems as you found them

A storage eval writes enormous scratch files onto shared filesystems that belong to other people. **Clearing them is part of the eval, not an optional courtesy** — an eval that hands back a report and leaves 165 GiB across five production mounts has cost the cluster more than it measured. The user should never be the one to discover the leftovers.

What this eval creates, all of which has to go:

| what | where | size |
| --- | --- | --- |
| `fio` work files | `<scan path>/fio-test/` on **every** mount scanned | ~33 GiB per mount per six-run scan |
| concurrent-write poke files | the share you wrote to, one per node | 1 GiB × nodes |
| `mamf-finder` / bench outputs, job logs | wherever you pointed them | small, but still yours |
| the isolated venv, cloned repos | `/tmp` or a share | leave only if under node-local `/tmp` |

Rules:

- **Confine scratch to one timestamped directory per mount** — `<mount>/users/<user>/cluster-eval-<name>-<TS>/` — so clearing it is one move per mount and nothing of the user's sits inside the blast radius.
- **Keep the results, clear the data.** The JSON / summary files are the deliverable; copy them off first (see the raw-artifact rule), then handle the multi-GiB work files.
- **Move, never `rm`.** On the node: `mkdir -p <mount>/users/<user>/trash/<YYYYMMDD>-<topic> && mv <scratch> <that dir>/`. Node-local `/tmp` is the exception — it evaporates with the pod.
- **A move within one Lustre mount is a rename, but a move across mounts copies every byte.** Keep the trash directory **on the same mount** as the scratch, or a "cleanup" will take longer than the benchmark did.
- Moving to trash on the same filesystem **does not free the space**. Say so, give the exact paths, and let the user purge — they are the only one who may destroy data.
- **Report the cleanup in your summary**: what you moved, where to, and how much. Then the user can empty it in one command.

## Isolated install

**Always build a fresh venv on the newest released PyTorch.** Never benchmark the node’s preinstalled env (conda, the image `python3`, a `dev` environment) — it is usually months behind. Torch version decides matmul kernels and NCCL version, so it changes MAMF and every `busbw` number. "It imports and the CUDA major matches" is **not** a reason to reuse it.

Find the newest, do not assume a version from memory:

```bash
python3 -m venv "$EVAL_VENV"   # $HOME/venvs/cluster-eval, or shared FS so every node sees it
"$EVAL_VENV/bin/pip" install -U pip
"$EVAL_VENV/bin/pip" index versions torch --index-url https://download.pytorch.org/whl/cu130   # -> LATEST: x.y.z
"$EVAL_VENV/bin/pip" install torch numpy packaging nvidia-ml-py matplotlib \
  --index-url https://download.pytorch.org/whl/cu130
```

Pick the `cuNNN` index matching the driver (`nvidia-smi`); try the highest one the driver supports and fall back if it 404s. Record in the report the installed version **and** that it was the latest on that index at eval time. If the newest wheel cannot be used (driver too old, no matching index), say so in Gaps with the version you settled for — do not quietly benchmark an old torch.

**Verify each phase actually ran on it.** Every bench prints its stack (`- software: torch=…, nccl=…`); MAMF logs it too. A launcher can silently substitute its own interpreter — `deepspeed`/`srun` run whatever `python3` is on the remote PATH, so a run can come back on the node’s old torch even with the venv sourced. Grep the log before you copy a number into the report.

**`nvidia-ml-py` is required** (`import pynvml`) — without it `mamf-finder.py` skips filtering of shapes that look fast because of cache artifacts (the script labels those SUSPECT), and MSMF can be inflated. The pip name is `nvidia-ml-py`; the import is `pynvml`.

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

| binary | package | why |
| --- | --- | --- |
| `ibstat` | `infiniband-diags` | InfiniBand/RoCE NIC (channel adapter) state, rate, LID. **Empty on EFA is OK** — then `rdma link` + sysfs rate |
| `fio` | `fio` | `fio-scan` |
| `rdma` | `iproute2` (+ `rdma-core`) | `rdma link` |
| `pdsh` | `pdsh` | DeepSpeed multi-node launcher / torchrun helper |
| `dcgmi` | `datacenter-gpu-manager-4-cudaN` (NVIDIA CUDA repo, not distro) | `dcgmi diag -r 2` hardware health |

If apt/dnf is blocked, record that in Gaps and fall back to `/sys/class/infiniband/*/ports/*/rate` + `nvidia-smi topo` — that is a last resort, not the default.

The five scripts from **Scripts (fetch)** must sit in **one directory** on a filesystem every node can read (shared FS), so `fio-scan` can call `./fio-json-extract.py`.

## Environment dump

Capture into the report (UTC datetime, hostnames):

```bash
date -u +%FT%TZ
hostname -s; uname -a
nproc; grep MemTotal /proc/meminfo
. /etc/os-release; echo "$PRETTY_NAME"
"$PY" -V
"$PY" -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.nccl.version())"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,power.limit,power.max_limit,clocks.max.sm,clocks.sm,power.draw,pcie.link.gen.current,pcie.link.gen.max --format=csv
nvidia-smi topo -m
# fabric (install infiniband-diags / iproute2 first — see Isolated install):
ibstat
rdma link
ls /sys/class/infiniband
cat /sys/class/infiniband/*/ports/1/rate
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv
sha256sum mamf-finder.py all_reduce_bench.py
```

Classify intra-node (NVLink vs PCIe from `topo`) and inter-node (InfiniBand / RoCE / EFA / Ethernet). HBM = `memory.total` reported in **GiB** (not MiB). SM clocks in the Environment GPU table: **boost** = `clocks.max.sm` (spec / short burst); **saturated** = SM clock at ~TDP under a dense GEMM (a range is fine; not a single nvidia-smi field); **parked idle** = `clocks.sm` when no compute. Do not label boost as “loaded” — a power-saturated GEMM sits well below boost. Do **not** name MAMF/MSMF in Environment — those terms start in Compute. Write the dump as **five tables** in the report: Host / GPU / Fabric / Filesystems / Software. Never one mega-table. **Software:** OS is **not** here — it is Host row 1. Python version + torch/NCCL. **Never** the eval venv path (temporal artifact). Tool versions (`mamf-finder`, …) belong under Compute.

## Compute

### Introduction

Open Compute with the **GPUs that are here** (name, count per node, SM count, HBM GiB, TDP, official BF16 TFLOPS + [table](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/README.md#tflops-comparison-table)). Do not start with DCGM or a FLOPS table. Then introduce the three passes as a **numbered list** (`1.` `2.` `3.`), each with the tool: `dcgmi diag -r 2` (NVIDIA's health check), [`mamf-finder.py`](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/benchmarks/mamf-finder.py) (shape search → **MAMF** burst *and* **MSMF** sustainable TFLOPS, both vs spec — do not describe it as burst-only), lemon GEMM (inline script). Headings below are **Hardware health**, **GPU performance benchmarks**, **Find underperforming GPUs** — not the tool nicknames.

### Hardware health

Hardware health, not a FLOPS number. `-r 1` is seconds of software/deployment only — skip it as a named step. **`-r 2`** is the light hardware suite (GPU memory, memory bandwidth, PCIe/NVLink; NVIDIA: ≲10.5 min on 8 GPUs). `-r 3` is a longer soak, optional.

Needs `nv-hostengine` and usually `CAP_SYS_ADMIN` (these trial pods are typically root with it). If install or diag fails, **Gaps** and continue with MAMF — do not skip the lemon pass.

```bash
nv-hostengine 2>/dev/null || true
dcgmi discovery -l
dcgmi diag -r 2 | tee dcgm-diag-r2.txt
```

One node is the minimum; extra nodes in parallel if you already have them. Report **Pass** or **Fail** per plugin and per GPU. If everything passed, write **Pass** — do not add **No Fail**. Copy the log off immediately.

### GPU performance benchmarks

Sequential MAMF — one GPU at a time, siblings idle:

```bash
for i in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=$i "$PY" -u mamf-finder.py --search auto --dtype bfloat16 \
    --output_file "mamf-gpu$i.txt" > "mamf-gpu$i.console" 2>&1
done
```

Optional concurrent all-8 is a **different** measurement (shared TDP/cooling); do not mix it into the per-GPU table. Compare MAMF/MSMF to official BF16 TFLOPS. Record per-GPU shape, power, SM clock. Flag any GPU >2% below the node median MAMF.

### Find underperforming GPUs

Cheap check for a slow, throttled, or dead card — **not** MAMF. Open with **TLDR: no under-performing or dead GPUs** or **TLDR: under-performing: gpu<N> on <node>**. One sentence: an under-performing GPU is slow or throttled, dead is the extreme case; same large `matmul` on every GPU, siblings idle. Run it on 1 node or 16 — the table has one row per node you have.

Fixed **16384³ bf16**. Compute SM coverage from **this** GPU’s SM count (`torch.cuda.get_device_properties(0).multi_processor_count`) — never reuse an SM count from another GPU family. At 128×256 output tiles that is **8192** CTAs (`16384/128 × 16384/256`; a CTA is one CUDA thread block). Full waves = `8192 // SMs`, tail = `8192 % SMs`, wave efficiency = `1 - tail/8192`. Example only: 132 SMs → 62 waves + 8 CTA tail (**99.9%**).

Show **every GPU’s TFLOPS**, not only min/max. **min** is the slowest score; **worst GPU** is which index hit that min (same number). **max** is the fastest sibling — the spread, not a substitute for the other scores. A card is under-performing when it lands well below its node median (MAMF used >2%); the min of a tight pack is noise.

## Network

Ask which NCCL/platform env vars to set. **Dump first** (`env | grep -E '^NCCL_|^FI_|^AWS_OFI'`). Keep platform values. Default **only if unset**: `NCCL_NVLS_ENABLE=2`. If `NCCL_TUNER_PLUGIN=ofi` (or similar), a **platform tuner** — not NCCL’s built-in `tuning.cc` — picks the algorithm; confirm with `NCCL_DEBUG`. `pdsh` does **not** copy the launcher’s environment — put overrides in `~/.deepspeed_env` **and** `source` them in the remote command, or embed `VAR=val` in the pdsh `cmd`. `--venv_script run-env.sh` is the DeepSpeed equivalent. DeepSpeed can drop exports from `--venv_script`; `~/.deepspeed_env` (one `KEY=VALUE` per line, in the launch cwd) is the reliable file. **`--venv_script` does not switch the Python interpreter** — DeepSpeed still launches whatever `python3` is on the runner PATH (often the node’s old install). Put the eval venv **first** on `PATH` inside `run-env.sh` *and* invoke `deepspeed` from that venv (`$EVAL_VENV/bin/deepspeed` or `PATH=$EVAL_VENV/bin:$PATH deepspeed …`). Confirm the table’s `torch=` / `nccl=` line matches the isolated install.

Connectivity first (`torch-distributed-gpu-test.py` — ranks must init NCCL and finish a collective). **The heading and TOC label must match the ranks you actually launched**: `Inter-node connectivity` only when it ran across every node, `Intra-node connectivity` for a single-node run. On ≥2 nodes use the same launcher as inter-node all-reduce (`deepspeed -H`, `pdsh`, …) with one rank per GPU; the snippet below is the 1-node form.

```bash
"$PY" -m torch.distributed.run --nproc_per_node="$NGPU" --rdzv_endpoint localhost:6000 --rdzv_backend c10d \
  torch-distributed-gpu-test.py
```

Intra-node (header of [`all_reduce_bench.py`](https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/all_reduce_bench.py)):

```bash
"$PY" -u -m torch.distributed.run --nproc_per_node="$NGPU" --rdzv_endpoint localhost:6000 --rdzv_backend c10d \
  all_reduce_bench.py
```

Inter-node (**≥2 nodes**, **all** of them). Let `NNODES` = hostfile length, `NGPU` = GPUs per node, ranks = `NNODES * NGPU`. **1 node: omit this whole subsection** — no heading, no “Skipped: 1 node”. **Do not discover a missing hostfile here** — that was the preflight gate. Use whatever launcher this cluster already has (`deepspeed -H <hostfile>`, `pdsh`, `srun`, or a site SSH wrapper). Reachability (SSH/TCP) already passed; this section is the GPU `all_reduce_bench.py` sweep only. Heading: `### Inter-node all-reduce benchmark` (ranks go in the TOC annotation, not the heading).

Hostfile: skip `#` comments. `MASTER_ADDR` = first **data** line, not a comment. If the platform already ships a hostfile, **read it, never edit it**.

**1. DeepSpeed** (from the shared cwd that contains the bench):

```bash
# run-env.sh: source the eval venv; re-export the NCCL values you dumped (do not invent 2 if the platform set 1)
time deepspeed -H "$HOSTFILE" --venv_script run-env.sh all_reduce_bench.py \
  |& tee all_reduce_bench-deepspeed.log
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
    $cwd/all_reduce_bench.py"

PDSH_RCMD_TYPE=ssh PDSH_SSH_ARGS_APPEND="-o StrictHostKeyChecking=accept-new -o BatchMode=yes" \
  pdsh -w $HOSTS $cmd |& tee all_reduce_bench-torchrun.log
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

`NNODES ≥ 2` `all_reduce_bench.py` `busbw` is **always** mixed: intra-node NVLink + inter-node NIC + (usually) NVLS/SHARP. Advertised per-GPU IB/RoCE/EFA GBps is **inventory**. It cannot be measured in isolation with this collective, NVLS on or off. NVLS is in-network all-reduce on the NVSwitch (~30% intra / ~25% inter vs ring `busbw`) and does **not** help other collectives.

Forbidden (do not score multi-node all-reduce as a fraction of NIC rate — that mixed NVLink+NIC+NVLS number is not “% of rail”):

- `busbw / 100 GBps` (or any NIC uni rate)
- `busbw * (k-1)/(n-1)` as “GBps on the wire” or “% of rail” — that factor is ring-traffic bookkeeping in [why inter-node busbw is not the wire](https://github.com/stas00/ml-engineering/blob/master/network/README.md#real-network-throughput), not an eval score, and it is wrong with NVLS (the default preset)

Quote instead: peak **busbw**, NCCL path (IB vs Socket; NVLS yes/no), NIC inventory on the Environment / transport line, and optionally multi-node busbw vs **1-node** all-reduce at the **same payload** (scale-out tax). Record `NCCL_NVLS_ENABLE` and other NCCL env.

Report template must include: transport + version, NIC inventory GBps, measured peak busbw, NCCL path — **not** “% of NIC spec.”

## Storage

### Introduction

Open Storage with **`### Existing Partitions` first**, then **`### Introduction`**. They are sibling headings, not nested. Never open with a fio table.

- **`### Existing Partitions`:** inventory only, as a **table** — `| mount | size | type | device / server |`. Deduplicate by **backing volume**, not by `df` line. Same RAID/device, same Lustre `nid@net:/fsname`, same CSI/PVC, or a bind-mount of another path = **one row**. Listing both `/` and `/tmp` (or two bind points of one share) looks like twice the disk. **Always include `/`** as the on-node-disk row, even when the local `fio-scan` ran under `/tmp` — `/tmp` is a scan path, not a second volume. **`type` is the backing filesystem** (`xfs`, `ext4`, `lustre`, …), not the container layer: `df -T /` often says `overlay` inside a pod — look through with `findmnt` / `lsblk -f` / the same-device `/tmp` mount and put that FS in the type column. Overlay is not a partition type. Order: `/` first, then one row per distinct shared volume, using the mount the cluster actually uses for work. Never loose paragraphs, never an “Also mounted” bucket. No `fio-scan`, no “this is the ceiling”, no “we poked this path”; nothing about what the eval did. Facts that apply to several rows (all one product, separate servers, a protocol shim to look behind) go in a sentence **after** the table. Do not bring in filesystem names from other clusters or past jobs.
- **`### Introduction`:** same convention as Compute and Network: a **numbered list of passes that ran**, each named by tool + path, not a list of mounts. GitHub-link [`fio-scan`](https://github.com/stas00/ml-engineering/blob/master/storage/fio-scan) on the first item and say what it does there (six `fio` runs over 16 KiB / 1 MiB / 1 GiB × read/write; 16 jobs, 4 KiB blocks, `libaio`, `O_DIRECT`; 3 min each). Later `fio-scan` mentions are bare `code`. Typical items: local `fio-scan` on `/tmp` (the baseline for the shared table's **local/shared ×** columns), shared `fio-scan` on the work FS, and — if **≥2 nodes** — concurrent 1 GiB write on a different share, **one writer per given node** (tool is `dd` unless you used something else). Omit the concurrent poke on 1 node.

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

`reports/<cluster>-<YYYY-MM-DD-HHMMZ>.md` plus `reports/raw-<name>/` (consoles, txt, fio json, plots), next to this skill or in the working directory. Example: `h200-2026-09-17-1938Z.md`.

**`### Gaps` under Compute, Network, or Storage:** omit the subsection entirely when that phase is complete. Never write `### Gaps` / `None`. Only include it when something is actually missing or incomparable (busy siblings, lemon GPU, DCGM `-r 2` failed/unprivileged, rdzv failed on ≥2 nodes, fio missing, …). **1 node is not a gap** — inter-node is omitted, not missing. When the rest of the report is written, **ask the user to address every remaining Gaps item**, each with a **proposed plan of action** (loop step 9).

```markdown
# Cluster eval: <name> (<N> nodes of <G>× <GPU>)

Example titles: `4 nodes of 8× H200`, `1 node of 8× B200`, `16 nodes of 8× H100`. Always include **node count and GPUs per node**, not only the GPU name.

- Datetime (UTC):
- Access:
- Nodes / GPUs:

Then a one-line pointer to the raw logs directory: `Raw logs: [\`raw-<name>/\`](raw-<name>/).`

## Table of Contents
Numbered list **before Environment**. Each item is **topic — tool/test that ran**, linking to that heading (GitHub-style `#anchor`: lowercase, punctuation stripped, spaces to `-`). Duplicate titles (`### Introduction`) get `-1`, `-2` suffixes; prefer linking the unique bench heading, not the intro. Only list tests that actually ran (plus inventory).

Compute TOC links are **purpose**, tools after the dash. Do **not** use DCGM / MAMF / Lemon GEMM as the TOC link text; those names live in the annotation or the section body. The `###` headings must match the TOC labels so the anchors agree.

The connectivity test has no section of its own — its result is item 1 of the Network `### Introduction`, so that TOC line links to `#introduction-1` (the second `### Introduction` in the document). Label it by the ranks it actually ran on: **Inter-node connectivity** only if it spanned every node, otherwise **Intra-node connectivity** with the rank count.

Network TOC: **Intra-node all-reduce benchmark** and **Inter-node all-reduce benchmark** — `all_reduce_bench.py`, then the rank count. Do not drop the word **benchmark**. Omit the inter-node line if 1 node.

Storage TOC: **Existing Partitions** — mount inventory. Then **Local disk IO benchmark** — `fio-scan` on `<local path>` and **Shared <product> IO benchmark** — `fio-scan` on `<shared path>` (same shape; path after `on`, never inside the link). Concurrent poke when ≥2 nodes: **Concurrent write shared fs benchmark** — `<N>-node 1 GiB O_DIRECT dd`.

Example:

```markdown
## Table of Contents

1. [Environment](#environment) — host, GPU, fabric, mounts, software (inventory, not a bench)
2. [Compute](#compute)
   1. [Hardware health](#hardware-health) — `dcgmi diag -r 2`
   2. [GPU performance benchmarks](#gpu-performance-benchmarks) — `mamf-finder.py` sequential, siblings idle
   3. [Find underperforming GPUs](#find-underperforming-gpus) — fixed 16384³ `matmul` on every GPU
3. [Network](#network)
   1. [Inter-node connectivity](#…) — `torch-distributed-gpu-test.py`, R ranks (label it **Intra-node connectivity** if it only ran on one node)
   2. [Intra-node all-reduce benchmark](#intra-node-all-reduce-benchmark) — `all_reduce_bench.py`, 8 ranks
   3. [Inter-node all-reduce benchmark](#inter-node-all-reduce-benchmark) — `all_reduce_bench.py`, 32 ranks (omit this TOC line and the section if 1 node)
4. [Storage](#storage)
   1. [Existing Partitions](#existing-partitions) — mount inventory
   2. [Local disk IO benchmark](#local-disk-io-benchmark) — `fio-scan` on `/tmp`
   3. [Shared Lustre IO benchmark](#shared-lustre-io-benchmark) — `fio-scan` on `/code`
   4. [Concurrent write shared fs benchmark](#concurrent-write-shared-fs-benchmark) — 4-node 1 GiB `O_DIRECT` `dd` (omit if 1 node)
5. [Findings](#findings)
   1. [Healthy subsystems](#healthy-subsystems)
   2. [Underperforming subsystems](#underperforming-subsystems) (omit if nothing underperformed)
   3. [Needs operator input](#needs-operator-input) (omit if every target was known)
```

Rebuild the anchors from the headings you actually wrote. Omit TOC lines for phases that did not run — no placeholder line saying they are missing.

## Environment
Five small tables — **Host / GPU / Fabric / Filesystems / Software**. Do not pile every field into one table. The third and fourth are **Fabric** and **Filesystems**, never “Network” / “Storage”: those would take the `#network` / `#storage` anchors the top-level sections need.

### Host
**First row: Ubuntu** (`PRETTY_NAME`, e.g. 24.04.2 LTS). Then hostname, kernel (`uname`), CPU cores/NUMA, host RAM, PCIe gen×width (`pcie.link.gen` / `width` from nvidia-smi — board/host path, not a GPU property). Not Python.

### GPU
name × count, SM arch / SM count, HBM in **GiB only** (not `275040 MiB (268 GiB)`), driver / CUDA, **three** SM clocks — boost (`clocks.max.sm`), saturated (~TDP under dense GEMM), parked idle — power limit / idle draw. Do not merge boost and “loaded”; a dense GEMM is not at boost. No MAMF/MSMF names here. PCIe does **not** go here.

### Fabric
intra-node fabric + spec uni GBps; inter-node adapter + rate as **inventory** (`ibstat` / `rdma link`).

### Filesystems
Left column **Local** / **Shared network FS** — do not repeat “local” or “ceiling” in the value. Local value: mount + size; if the backing device is Linux md, write `md127 RAID` (or whatever `mdN`), not a bare `md127`. Shared: guest-visible type + path + size; add the vendor product name when it is not the same as `df -T`. **Capacity only — never how much is used or free.**

### Software
Python version, torch, NCCL (`torch.cuda.nccl.version()`). Note that torch was the newest release at eval time. **No eval venv path.** Isolated venv is how you run, not a report field.


## Compute

### Introduction
Name × count, SM count, HBM GiB, TDP, official BF16 TFLOPS. Not a benchmark dump. Then a **numbered list of the passes that follow**, each saying what the tool literally does:

1. `dcgmi diag -r 2` — NVIDIA's own health check (software, memory, PCIe).
2. `mamf-finder.py` — times many **matmul shapes** (M×N×K) per GPU to find the fastest; reports **MAMF** and **MSMF**, both vs the official spec.
3. Lemon GEMM — a throwaway inline script: one fixed matmul shape (M=N=K=16384) on every GPU, to catch a dead, slow, or throttled card.

### Hardware health
**Pass** (or **Fail** on named plugins/GPUs; or Gaps: no hostengine / unprivileged). Never **Pass** and **No Fail** together.

### GPU performance benchmarks
| GPU | MAMF | MSMF | MAMF shape | MSMF shape | MAMF W/MHz | MSMF W/MHz | MAMF % spec | MSMF % spec |
min / median / max across GPUs. Official TFLOPS source = [TFLOPS comparison table](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/README.md#tflops-comparison-table), not a relative path. Open that section by explaining the measurement before any number: compute is measured by timing matrix multiplication; how fast a `matmul` runs depends on its **matmul shape** (the M×N×K dimensions), so the tool searches — state how many matmul shapes it timed per GPU, as a range if the `auto` search tried a different count on each card (read the `Tried N shapes` line in every `mamf-gpu*.txt`, don't extrapolate from one). Then `Two numbers per GPU vs official BF16 <N> TFLOPS on <GPU>:` (name the GPU; do not leave the spec floating), then two bullets, each with the measured W/MHz the script printed:

- **MAMF** (Maximum Achievable Matmul FLOPS) — highest TFLOPS reached by any matmul shape (short burst at boost, board far under its power limit). A ceiling, not a rate any sustained workload holds.
- **MSMF** (Maximum Sustainable Matmul FLOPS) — best that survives once the matmul shape is large enough to pin the board at the power limit and the clock settles. Compare a sustained workload to this number.

**Expand every acronym at first use in the report** — MAMF, MSMF, DCGM, NVLS, GEMM, busbw/algbw, MFU. The Terms table in this skill is for the agent; the reader of the report only sees the report.

**The eval does not know what the cluster is for.** Never assume training — the same allocation may be bought for inference, fine-tuning, or batch jobs. Write "sustained workload", not "training step"; say "training or inference" when you need an example.

After the bullets, say what the table's shape and `W/MHz` columns are: the M×N×K that won each regime, and the board power / SM clock measured while it ran.

### Find underperforming GPUs
**TLDR:** `no under-performing or dead GPUs` — name both outcomes, since a dead card and a slow one are different findings. If something is wrong, say which: `under-performing: gpu<N> on <node>` or `dead: gpu<N> on <node>`. Then define an under-performing GPU (slow or throttled; dead is the extreme case) and the pass: the **same** matmul shape (M=N=K=16384) on every GPU, siblings idle, flag a card well below its node's median — explicitly **not** MAMF (no search over matmul shapes, no boost/sustain split). Then 16384³ SM coverage. Table: all GPU scores per node you have, min (gpu), max. Do **not** prefix the heading with node count — ranks belong in the TOC annotation, not the heading.

## Network

### Introduction
Open with a **numbered list of tools** (same shape as Compute and Storage), each explained before any number: GitHub-link [`torch-distributed-gpu-test.py`](https://github.com/stas00/ml-engineering/blob/master/debug/torch-distributed-gpu-test.py) (connectivity — one process per GPU; ranks see each other and complete a collective; say how many ranks and over how many nodes it ran, and do not call a 1-node run inter-node) and [`all_reduce_bench.py`](https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/all_reduce_bench.py) — say what it sweeps (payloads from 32 KiB to 16 GiB) and define `busbw` (bus bandwidth, what the fabric moves) and `algbw` (algorithm bandwidth, what the caller sees) **there**, plus which rank counts it ran at. Do not write “the book” or “MLE”. Then NVLS (see Terms): in-network **all-reduce** on the NVSwitch (~30% intra / ~25% inter vs ring `busbw`); does **not** help other collectives. InfiniBand switch SHARP is a different feature. Record `NCCL_NVLS_ENABLE`: **2** = auto — the eval default if unset; **0** = force off; **1** = require NVLS (can abort if unavailable). State whether NVLS was available for intra and for inter (`NCCL_DEBUG` when you have it). Name who set the env (container image / cluster), not that you “kept” it.

### Intra-node all-reduce benchmark
Intra-node: transport (e.g. NVLink 5, NV18), spec uni GBps, busbw table, peak % spec (note NVLS if on).
**Inline** `![Intra-node all-reduce](raw/<cluster>/busbw-mean-…-8.png)` next to that section.
### Inter-node all-reduce benchmark
Inter-node **only if ≥2 nodes**: transport, NIC inventory GBps, table, peak busbw vs 1-node at the same payload, NCCL path (not % of NIC spec). Do not name the launcher. Rank/node counts go in the TOC annotation. **1 node: delete this subsection and its TOC entry.**
**Inline** `![Inter-node all-reduce](raw/<cluster>/busbw-mean-…-R.png)` when the plot exists.
### Gaps
(only if needed, e.g. rdzv failed on ≥2 nodes — not “only 1 node”)

## Storage

### Existing Partitions
Inventory table: **always a `/` row**, then one row per **distinct** volume. Same backing store mounted twice (`/tmp` on `/`, bind-mount aliases) is still one row. No benchmarking comments. Do not name a network FS this allocation does not use. Shared facts go after the table.

| mount | size | type | device / server |
| --- | ---: | --- | --- |
| `/` | 28T | xfs | md127 RAID0 (8× ~3.5T NVMe) |
| `/code` | 2.3T | Lustre | `10.4.138.63@tcp:/xmym3bev` |

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

Heading is **Shared <product> IO benchmark** (e.g. Shared Lustre IO benchmark), not a bare “Shared”. Optional extra when ≥2 nodes: **Concurrent write shared fs benchmark**. Omit on 1 node.

Local NVMe vs shared network FS (16 KiB / 1 MiB / 1 GiB, read+write). bw = MiBps (`_` on the fast table). IOPS = 10⁶ in every table. **`size` is right-aligned** (`---:`), like every other numeric column; only `rw` is left-aligned. Each shared table appends **`local/shared ×`** at the **end** (local bandwidth ÷ shared bandwidth) so the first five columns match local. Explain the column in a **sentence**, not a chain of equations: what it divides and what 1 would mean. Do not add an IOPS ratio: fixed 4 KiB blocks make it identical to the bandwidth ratio. **Quote table values verbatim** in prose — if the column says `26.5`, write `26.5×`, never `27×`. Do not express the comparison as a percentage, do not call it `%local`, and say "the local disk table above", not a device name. Raw unscaled numbers in `raw/`.
(omit Gaps unless something is missing)

## Findings

Up to three headings, in this order — **### Healthy subsystems**, **### Underperforming subsystems**, **### Needs operator input** (the noun is required; never a bare `### Healthy`). Include only the ones that have bullets. Bullet labels are the subsystem (GPUs, fabric, local disk, shared network FS, torchrun) — **never prefix with A./B./C.** (the report sections are Compute, Network, Storage). Split mixed areas (e.g. local disk vs shared FS). Do not put an underperforming item in Healthy with a caveat, and do not hide it in a Flags list that looks like the healthy bullets.

Healthy subsystems: fabric and GPUs that met spec. Network headline is **fabric** (intra vs NVLink spec + NVLS; inter busbw + path + vs 1-node all-reduce) — **never** “X% of NIC/rail,” and **not** a launcher. Do not credit DeepSpeed / torchrun / srun in busbw or Findings.

Underperforming subsystems: measured vs **vendor advertised** for **this** product; not good enough for code (16 KiB) / checkpoint (1 GiB → 1 TiB) / dataloader; torchrun/rdzv if it failed; leftover Gaps. A broken launcher is **not** “the network is unhealthy.” Do not name a network FS this allocation does not use.

**Needs operator input: a verdict requires a target, so a measurement without one goes here, not under Underperforming.** Calling a subsystem slow when you have nothing to compare it against is an opinion dressed as a finding — the reader cannot act on it and the operator can dismiss it. This heading is for a subsystem that was measured cleanly but whose expected figures only the operator has: a shared FS with no published provisioned throughput, a fabric whose purchased tier is unknown, any device whose spec sheet does not exist publicly. Open with one sentence saying these cannot be called healthy or underperforming until those figures arrive. Each bullet gives the measurements in full, then **what to obtain**.

**Write targets in the plural — a filesystem never has one.** Every file size and operation has its own rate, reads and writes differ, so it is "the figures they are supposed to hit" and "no published performance figures", never the singular. The singular quietly promises the reader one number to compare against, which is the mistake the whole category exists to avoid.

**Ask only about the mount you measured, and name its size.** Expected rates are per filesystem, so asking for "`/code` and `/data`" when six numbers came from `/code` alone invites an answer you cannot use. State which mount the figures describe, and if sibling mounts differ in size, say that each has its own target and needs its own scan — pointing at the one the workload will lean on hardest.

**Ask for the targets in the same shape as your measurement, or they will not settle anything.** "The provisioned throughput" is not askable — a parallel FS has no single rate, and a per-TiB headline number cannot be checked against a row of the table. Ask for **one expected value per measured cell, in the units you measured, under the conditions you measured**: for `fio-scan` that is bandwidth in MiB/s **or** IOPS at 16 KiB, 1 MiB and 1 GiB, read and write, from one client with 16 jobs, 4 KiB blocks and `O_DIRECT`. Either unit is enough because the fixed block size makes one derivable from the other. Say those conditions in the ask, because the same filesystem answers differently per client and in aggregate, and because small-file and large-file rates come from different limits — one headline figure hides both. Then add the conditions that would invalidate a comparison if the operator's figures assume something else: per-client vs per-filesystem, and stripe count. A comparison against another subsystem you did measure (shared FS vs local disk, `26.5×` slower) is a fact and belongs in the bullet; it is not a substitute for the target, because local NVMe was never what the shared FS was sold to match.

Judge before you place a bullet: is there a number this is supposed to hit? Official TFLOPS, NVLink spec GBps and a vendor-advertised FS figure are targets, so those bullets can be Healthy or Underperforming. If the answer is "the operator would have to tell me", the bullet goes in **Needs operator input** even when the measurement looks disappointing.

**Quote the whole measurement, compactly, not one cherry-picked row.** A storage bullet has six numbers behind it, so give all six in one line — `read / write: 16 KiB **a MiB/s / b MiB/s**, 1 MiB **c MiB/s / d MiB/s**, 1 GiB **e MiB/s / f MiB/s**` — instead of a single size, which leaves the reader asking what the others were. Same for any bench that swept a parameter. Compact, not partial.

**Every number outside a table carries its unit.** Putting `MiB/s` once in the lead-in and then listing bare `1_646.1 / 1_658.3` is a table habit; in prose each value needs its abbreviation — `1_646.1 MiB/s / 1_658.3 MiB/s`. `read / write` once is fine. Same for GBps, TFLOPS, msec, ×. The table header is the one place a unit may appear only once.

**When vendor figures are missing, say the targets are unknown, say why none are published, and ask for them — but do not narrate the search.** Three things, because a measurement with no target cannot be called a shortfall, and a reader who knows the filesystem's name will reasonably assume a spec sheet exists: (1) state plainly that the provisioned figures are unknown and the measured numbers therefore have nothing to be judged against; (2) explain in a sentence why — for a parallel FS, **the software publishes no performance figures**; Lustre, GPFS/Storage Scale, BeeGFS and NFS are filesystem software, and throughput is a property of the deployment (count of object storage targets / servers, their backing devices, the network to them, per-file stripe count), so the number that exists is the **service tier the operator purchased**, typically sold per TiB of capacity (managed Lustre commonly 125–1000 MB/s per TiB); (3) ask the user for that tier **and** the default stripe layout, since a single-stripe file is served by one storage target regardless of filesystem size and is the usual explanation for a single client seeing a few hundred MiB/s. What to leave out is the search: where you looked, what `df` or the mount table did not contain. A mount was never going to state IO speed, so ruling it out tells the reader nothing and reads as an excuse.

**Say who is slowed and by how much, in literal terms.** Hardware and files do not "feel", "suffer", "struggle", "get punished", or "care"; a filesystem is not "painful". Name the concrete operation and attach the measured rate to it — "a `git clone` or `pip install` on `/code` runs at the 16 KiB read rate of 204.7 MiB/s" — so the reader can check the claim against the table instead of trusting an adjective.

If nothing underperformed, **omit that heading and its TOC line** — never write `Underperforming subsystems: none`. Same for **Needs operator input** when every target was known. No extra **Flags** section.
```

## Final pass before you show the report

Prose rules get skipped. **Run this list against the finished report, line by line, before you tell the user it is ready.** Each item has failed a real review.

| check | fix |
| --- | --- |
| Every acronym expanded at first use (MAMF, MSMF, DCGM, NVLS, GEMM, busbw, algbw) | `Maximum Achievable Matmul FLOPS`, … |
| Every tool says what it literally does before its first number | how many shapes / payloads / runs, and what it varied |
| No term used before it is defined ("any shape", `local/shared ×`, CTA, `O_DIRECT`) | define at first use, in a sentence — not as an equation |
| Shared-vs-local stated as **times slower**, not percent | `8.0×`, not `12.4%` |
| Prose numbers identical to the table cell they cite | table `26.5` → prose `26.5×`, never `27×` |
| Findings bullets carry the full sweep, compactly | all three sizes × read/write on one line, not just 1 GiB |
| Every number outside a table has its unit | `1_646.1 MiB/s / 1_658.3 MiB/s`, not a lead-in `MiB/s:` then bare numbers |
| Nothing inanimate given feelings | not "small files will feel that" — name the operation and its measured rate |
| Targets in the plural | "the figures they are supposed to hit", never "the figure" |
| Missing vendor figures: target-unknown admission + why none are published + the ask, no search narrative | keep "provisioned throughput is unknown, so there is no target to compare against"; drop "the mounts show only the server address and size" |
| No bullet judged without a target | no known figure to hit → **### Needs operator input**, not Underperforming |
| Shared-FS figures scoped to the mount they came from, with its size | one mount's table is never "the shared FS"; differently-sized siblings each need their own scan |
| Distinct shared mounts grouped only by the 5% rule | one table only if every corresponding bandwidth and IOPS cell is within 5%; otherwise separate tables |
| The ask names one expected value per measured cell, with conditions | "expected MiB/s or IOPS at 16 KiB / 1 MiB / 1 GiB, read and write, one client, 16 jobs, 4 KiB blocks, `O_DIRECT`" — not "the provisioned throughput" |
| Sizes in IEC units | `16 KiB` / `1 MiB` / `1 GiB`, never `16k` / `1m` / `1g` |
| Storage table headers carry units in `()` | `size`, `rw`, `latency (msec)`, `bw (MiBps)`, `IOPS (M)` |
| Numeric columns right-aligned, including `size` | `---:` for all but `rw` |
| Every `###` heading matches its TOC label, and every anchor resolves | rebuild anchors from the headings you wrote |
| Labels match what actually ran | a 1-node connectivity run is **intra**-node |
| No absence asserted | no `Skip reasons: none`, `### Gaps` / `None`, `N/A` |
| No redundant verdicts | `Pass`, not `Pass … No Fail` |
| No filler sentences | "collects the results into one table", "as expected" |
| No workload assumed | "sustained workload", not "training step" |
| Scratch cleared from every mount you wrote to | `du -sh` each scan dir; `fio-test/` moved to that mount's `trash/`, paths and sizes named in your summary |

Then fold any correction the user still makes back into this file (see **Keep this skill up to date**).

## Training wheels

These are **examples of past jobs**, not defaults and not required knowledge. Discover the next cluster from scratch (how to exec, namespace, launcher, NCCL env, mounts). Skip this section if none of it matches.

- One AWS EFA allocation used `kubectl` in namespace `mltraining-dev`, hostfile `/data-fast/hostfile`, SSH port **7878**, Lustre on `/code` `/data` `/checkpoint`, and image env `NCCL_NET_PLUGIN=ofi`, `NCCL_TUNER_PLUGIN=ofi`, `NCCL_NVLS_ENABLE=1`. `ibstat` was empty; `rdma link` showed `rdmap*`.
- One other trial used virtiofs on `/mnt/data`, container `sshd` **2222**, and hostname → `127.0.1.1` (torchrun needed `is_host` + `local_addr`).

Lessons:

- `pip install matplotlib` in the eval venv on **shared FS** so **every** node can plot. `all_reduce_bench.py` writes the PNG in the **process cwd** (often `$HOME` under torchrun), not the eval directory — copy it immediately. If a launcher still uses a different interpreter without matplotlib, regenerate the PNG from the printed table on the laptop.
- Environment is for a reader who has not seen MAMF yet: no MAMF/MSMF, no eval venv path, HBM in GiB, PCIe on Host, OS first Host row, Python on Software, Local cell must not repeat “local”/“ceiling”, md devices as `md127 RAID`.
- Network opens with **### Introduction** (NVLS + who set `NCCL_NVLS_ENABLE`), not a bare env-var line.
- Find underperforming GPUs: TLDR first (**no under-performing or dead GPUs**); all GPU scores; 16384³ SM coverage **from this SM count**.
- Install `infiniband-diags` (`ibstat`), `fio`, `iproute2`/`rdma-core` (`rdma link`), `pdsh`, and DCGM **before** the env dump. Use `sudo` when not root. Empty `ibstat` on EFA is OK if `rdma link` works. If `dcgmi diag -r 2` cannot run, Gaps — do not skip MAMF.
- Shared-FS `fio-scan` 1 GiB × 16 jobs × `O_DIRECT` **can** take hours (file create/unlink on some network FS). A quiet 40 min is not necessarily a hang. Copy logs after the local-disk phase.
- Retry pip/`kubectl exec` on connection reset; do not treat RST as a failed install until a second probe (`import matplotlib`, `command -v fio`, `command -v ibstat` or `rdma`).
- `set -e` + `grep` with no matches aborts scripts (busy-GPU check). Use `grep -c … || true`.
- Superseded artifacts go to a local `trash/<YYYYMMDD>-<topic>/` directory, never `rm` — wait for the user if the skill environment forbids delete.
