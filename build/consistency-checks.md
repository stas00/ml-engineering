# Consistency Checks (maintainer runbook)

This is a runbook **for a code agent** (or a human maintainer). It is not part of the published book and is not listed in `chapters-md.txt`.

Goal: periodically re-run a set of editorial/QA consistency checks over the book and fix any regressions. Read this file, then perform each check below in order, applying the decision rules and reporting/fixing findings.

All commands assume the repo root as the working directory. Use `rg` (ripgrep) for searching. Paths in this doc are relative to the `ml-engineering` repo root unless noted; the companion book **The Art of Debugging** (AoD) is assumed to live at `../the-art-of-debugging`.

Agent instructions:
- Work check by check. For each match, **classify before editing** using the rules given — do not blind-replace.
- Prefer targeted string edits. Never rewrite whole files.
- Leave literal command/API/env values, product-SKU names, and verbatim third-party tool output untouched (see rules).
- When a judgment call is genuinely ambiguous (e.g. a memory *footprint* vs *capacity*), flag it rather than guessing.
- At the end, report a concise summary of what was changed and what was intentionally left.

---

## Check 1 — Byte-unit consistency (GB vs GiB, MB vs MiB, ...)

Find every numeric byte-unit token:

```bash
rg -n '[0-9]\s?[KMGTPkmgtp]i?[Bb]\b' --glob '*.md' --glob '*.py'
```

Classify each hit and normalize per this table:

| Context | Unit | Examples |
| --- | --- | --- |
| On-device memory **capacity** (VRAM, CPU RAM, SRAM, on-chip cache) | binary `KiB/MiB/GiB/TiB` | "H100 has 80GiB", "640GiB of GPU memory", "256MiB cache", "1-2TiB of CPU memory" |
| Any quantity **computed via `2**n`** (or reported by a tool that divides by `2**n`/`1024**n`) | binary | activation memory `.../2**30`, `torch.cuda.mem_get_info`, `see_mem_usage` output, RSS `/2**20` |
| Benchmark **payloads that are `2**x`** | binary | `all_reduce_bench.py` sizes (`32KiB..16GiB`); it appends `iB` via `fmt_bytes` |
| **Bandwidth / throughput** | decimal `GB/s`, `GBps`, `Gbps`, `TBps` | NVLink/IB/EFA rates, busbw/algbw columns |
| **Network transmission volumes / payloads matched to decimal bandwidth** | decimal `GB` | "send 320GB over the wire (`80*4`)", ZeRO/DDP "60GB of data", `all_reduce_latency_comp.py` (`/1e9`) |
| Model memory **footprints** written as `params × bytes` with clean decimal arithmetic | decimal `GB/MB` (**leave**) | inference "`8B × 2 = 16GB`", "`2B × 18 = 36GB`", KV-cache "`/10**6 = 0.131MB`" |
| **Disk / storage** capacity & usage, on-disk **file sizes** | decimal `GB/TB` | "2.3TB checkpoint", "100TB tier", "2TB SSD", "1.2GB model file", core file "5GB" |
| **I/O block/file sizes** in an inherently binary context (fio) | binary | "block size of 4KiB", "16KiB Python files" |
| Item **counts** (see Check 2) | bare `K/M/B` | "10K samples", "8B params", "50k vocab" |

**Never touch** (leave exactly as written):
- Product-SKU names: `A100 80GB`, `A100-SXM4-80GB`, `H100 80GB HBM3`, `MI300X 192GB`, `v100-32g` partition, etc.
- Literal CLI/API/env values: `dd bs=1G`, `mount -o size=1G`, `systemd-run -p MemoryMax=5G`, `MEMLIMIT=5GB`, `max_shard_size="2GB"`, `nccl-tests -b 32k -e 16G`, `--shm-size=1g`, `3<<10`.
- Verbatim third-party tool output: `ls -lh` sizes (`304K`, `5.8M`), `df -h`, `ifconfig` (`138.4 GB`), `nvidia-smi`, `rocminfo` (`4KB Alloc Granule`), PyTorch OOM messages (already emit `GiB`/`MiB`).

**Author's own scripts:** when a script the author maintains prints a mislabeled unit (e.g. divides by `2**30` but prints `GB`), fix the label in the script too (e.g. `see-mem-usage.py`, `torch-dist-mem-usage.py`, `all_reduce_bench.py`). After editing any `*.py`, `python3 -m py_compile` it.

Quick spot-check that the `see_mem_usage` (`[0] mp:`) output has no stale `GB`:

```bash
rg -n 'mp:.* GB\b' --glob '*.md'
```

To sanity-review what non-binary byte tokens remain (each should be an intentional decimal/SKU/verbatim case):

```bash
rg -n '[0-9]\s?[KMGT]B\b' --glob '*.md'
```

---

## Check 2 — Bare `k` / `M` qualifiers

```bash
rg -n '[0-9][kKmMgG]\b' --glob '*.md' --glob '*.py'
```

Rule: a bare `K`/`M`/`B` is allowed **only when it counts items** (tokens, parameters, samples, vocab entries, ports, lines, GPUs, dollars). If the number denotes **bytes**, give it a real unit per Check 1 (e.g. prose "64M file" → "64MB file").

Leave bare: parameter/token counts (`175B`, `125M`, `10K` params, `250k` vocab, `3k` tokens), dataset names (`openwebtext-10k`, `ADE20k`), literal command sizes (`bs=1G`, `-e 16G`), verbatim `ls`/`df` output, and raw numeric values (`fp16` max `64K`, `64k` ports, `20k` scrollback lines).

---

## Check 3 — Cross-book sync with The Art of Debugging (AoD)

Some chapters are shared/overlapping between this book and `../the-art-of-debugging`. Keep their **content** in sync while preserving each book's **conventions**.

Known shared content:
- `debug/pytorch.md`  ↔  `../the-art-of-debugging/pytorch/README.md` (near-identical)
- The "emulating out of resources" memory one-liners in `debug/*` overlap with AoD `methodology/README.md`.

Diff the shared PyTorch chapter:

```bash
diff ml-engineering/debug/pytorch.md ../the-art-of-debugging/pytorch/README.md
```

When reviewing the diff, **sync genuine content** (prose wording, numbers, typos, unit fixes) but **do NOT "fix"** these intentional per-book differences:
- **Links** — this book uses relative paths internally and absolute `github.com/stas00/the-art-of-debugging/...` URLs for cross-book links; AoD does the mirror.
- **Heading/label case** — AoD uses sentence-case headings and lowercase labels (`note:`, `important:`, `tldr:`); this book uses Title Case / `Note:` / `Important:`.
- **Code-fence language tags** — AoD tags fences (```` ```python ````/```` ```bash ````); this book often leaves them bare.

A filtered diff to confirm no *unit/content* drift remains:

```bash
diff <(grep -n . ml-engineering/debug/pytorch.md) \
     <(grep -n . ../the-art-of-debugging/pytorch/README.md) \
  | rg -i 'GB|GiB|MiB|see_mem|mp:'
```

Also apply Check 1 + Check 2 to the AoD book itself (`../the-art-of-debugging/**/*.md`); the same rules apply. Note AoD's memory-testing one-liners in `methodology/README.md` allocate via `x 2**30` and display via `/2**20` → those are `GiB`; its `dd`/`tmpfs`/`systemd`/`MEMLIMIT` values are literal and stay as-is.

---

## Check 4 — Internal links & anchors

Every in-repo link target and `#anchor` should resolve. GitHub anchor slugs are produced by: lowercase; strip punctuation except `-` and spaces; spaces → `-`; de-duplicate collisions with `-1`, `-2`, ...

Fast build-based check (renders HTML then validates local links):

```bash
make check-links-local   # builds html-local, runs linkchecker on local links
```

If a build isn't desired, scan manually: collect all `](...)` targets and `[...](#anchor)` fragments per file, compute the slug set from that file's headings, and report any target file/anchor that doesn't exist. Fix by correcting the path/anchor (never by deleting the link silently).

---

## Check 5 — External link liveness

```bash
make check-links-all     # includes --check-extern; output in linkchecker-all.txt
```

Review `linkchecker-all.txt`. Beware **false positives** — these are almost always fine even when reported as errors:
- GitHub returning `404`/`429` to bots (rate-limiting / anti-scraping) for `tree/`/`blob/`/`issues/` URLs.
- `418 I'm a teapot` from freedesktop.org, timeouts from gnu.org, and similar anti-bot responses.

Only fix a link if it is **genuinely dead** (domain gone, page permanently moved). Verify by fetching in a browser/`WebFetch` before changing. Prefer an authoritative replacement (vendor/wiki/official docs, or the source in a GitHub repo if the docs site is decommissioned).

---

## Check 6 — Numeric fact-check

Spot-check technical claims — accelerator specs (TFLOPS, HBM capacity/bandwidth, pin rates), interconnect rates (NVLink/PCIe/InfiniBand/EFA), memory formulas (weights/optimizer/gradients/activations/KV-cache), and collective-comm math (all-reduce `2(n-1)/n`, etc.).

Method:
- Verify each number against a **high-accuracy source** (vendor datasheet, JEDEC, Wikipedia) — not random blogs.
- Check **internal consistency**: e.g. per-port vs per-node vs cluster aggregate should multiply out; `GiB × count` totals should add up; a value converted to `Gb`/`Gbps` (`×8`) must match.
- Watch unit correctness (Check 1) and distinguish physical vs usable capacity (e.g. B200 192GiB physical vs 180GiB usable).
- **Flag before fixing**: for fact changes, present the finding (file:line, current value, proposed value, source) and let the maintainer confirm, since some values are deliberately approximate.

Files historically densest in checkable numbers: `compute/accelerator/README.md`, `network/README.md`, `network/comms.md`, `storage/README.md`, `training/performance/README.md`, `training/model-parallelism/README.md`, `inference/README.md`.

---

## Reference: unit conventions in one line

Capacity & `2**n`-computed quantities → binary (`GiB`); bandwidth, transmission volumes, disk, on-disk file sizes, and clean `params×bytes` footprints → decimal (`GB`); SKU names / literal command args / verbatim tool output → leave; bare `K/M/B` only for item counts.
