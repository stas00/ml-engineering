# Session Preferences

## Change approval

1. Always present a concrete proposal before editing source content.
2. Do not apply a proposed correction, addition, or table change until the user explicitly approves it.
3. A request phrased `do N` treats suggestion `N` and any qualifiers in the request as the concrete approved proposal for immediate implementation; do not request a second approval.
4. A request phrased `let's look at N` requests inspection and discussion only; do not edit until the user later approves a proposal.
5. During consistency reviews, produce suggestions rather than silently fixing content. Internal anchor corrections were the only stated exception during the initial review.
6. Never delete or remove any existing content, data, row, note, file, or roadmap entry without first discussing the exact proposed deletion and receiving explicit user approval.
7. Propose deletion only when there is a concrete justification such as invalid, incorrect, harmful, or genuinely redundant content; restructuring, cleanup, uncertainty, or lack of a citation is not sufficient.
8. Unverified, rumoured, or unofficial information is not a deletion candidate - it is often exactly what a reader can't get from a vendor page, and removing it because it lacks a citation destroys the most valuable kind of content in the book. Label it as unconfirmed and say what would change if it held, rather than dropping it. "I could not verify this" is a reason to annotate, never a reason to delete.
9. Before any deletion, explain the reason, scope, consequences, and preservation alternatives so the user can make an informed decision.
10. If restructuring makes existing data difficult to retain in its original location, leave it unchanged and propose a relocation, annotation, or schema adjustment.
11. When replacing or restructuring a table, account for every original row and explicitly flag any row that cannot be represented faithfully.

## Book style

1. Before proposing or writing content, inspect the surrounding section and representative existing sections of the book for established conventions.
2. Follow existing local style for voice, terminology, heading depth, list markers, notes, citations, tables, and source layout rather than introducing a new pattern.
3. When proposing a fix, propose the smallest change that fully corrects the problem. Preserve as much of the original text as possible and avoid rewriting correct surrounding prose merely for polish.
4. Preserve the original author's unique voice, energy, and charm. Correct mistakes without flattening the writing into generic technical prose; broader stylistic rewrites require explicit approval.
5. Prefer the style of the file and nearby section when book-wide usage varies.
6. An explicit user-approved format, such as numbered per-table source lists, overrides a conflicting precedent.
7. If existing style cannot express the content clearly or is inconsistent enough to create a maintenance problem, flag the problem and make a numbered improvement suggestion before changing the style.
8. Reader experience is the first priority when proposing changes. Start with the reader's task: what the reader needs to understand, compare, decide, or do.
9. A technically correct proposal is incomplete if it makes the material less useful, harder to navigate, or harder to apply. Among correct alternatives, prefer the one that best preserves or improves the reader's workflow.
10. For comparison tables, define the comparison question and explain how the reader should use the values before proposing columns, normalization, or vendor-specific qualifications.
11. Make comparison results visible immediately. Precompute useful ratios, deltas, rankings, or normalized values instead of requiring the reader to perform arithmetic.
12. Lead with the reader-facing conclusion or at-a-glance comparison; place provenance, vendor terminology, derivation details, and caveats close by as supporting information rather than making them the primary interface.
13. The book is written in English, so never add text in another language to a chapter. Consulting a non-English source is encouraged - it is often the only place a specification is published, as with Huawei's Chinese Ascend pages carrying per-accelerator figures the English pages omit - but what lands in the chapter is the English translation, not the original string. Do not paste the source text alongside the translation either: a reader who can't read it gains nothing, and it makes the line harder to scan.
14. Translate the meaning rather than transliterating, and normalize the numbers to the book's own [Unit formatting](#unit-formatting) conventions, since a translation is no longer a verbatim quote and the vendor-quote exception no longer applies. Romanize a proper name that has no English form and gloss it once, as with `LingQu` for Huawei's UnifiedBus fabric.
15. Say in prose where a figure came from when the source is non-English, so the reader knows the claim is traceable and knows which site to check. Naming the language is useful; reproducing it is not.

## Reader-visible grounding

1. Every number, name, command, or claim that lands in a chapter must be derivable from what the reader can see: the same section, earlier content in the same chapter, or a cross-linked chapter or file in the repository.
2. Never lean on anything that exists only in the assistant's context - benchmark output pasted into chat, a command run while researching, a file read but not quoted, a figure computed in a previous turn. The reader has none of it, and the resulting text looks rigorous while being impossible to check.
3. When a derivation needs a value the chapter does not show, there are exactly two acceptable moves: add the value to the chapter, or restructure the derivation so it uses values that are already there. Silently using the hidden value is the failure mode.
4. This applies with equal force to intermediate steps. A worked example that passes through an unshown quantity is unverifiable even when its final answer happens to be right.
5. Prefer the derivation the reader can redo with the numbers in front of them, even when a hidden value would be more direct.
6. Concrete failure: the `Inter-node speed depends on intra-node speed` section derived a 4GiB elapsed time from an `algbw` figure that came from raw benchmark output pasted into chat, while the section's own table shows only `busbw`. The fix was to recover the time from `busbw` by undoing the correction factor the chapter's glossary already defines.
7. When quoting or computing from a script's output, adopt the script's own unit definitions rather than assuming them. `all_reduce_bench.py` prints `1GiB = 2**30 Bytes` alongside `1GBps = 10**9 Bytes per second`, so a `GiB / GBps` division needs the base conversion. Treating them as one base understated every elapsed time in that section by about 7%, and the error survived a full review pass because each individual figure looked plausible.

## Measured, derived, or assumed

1. Every quantity in a chapter is one of three things: measured, derived from measurements, or assumed. The words around it have to say which. `measures 73%` is a claim about an instrument and `works out to 73%` is a claim about arithmetic - they are not interchangeable.
2. The arithmetic being correct is not what is under review here. Each failure below had sound arithmetic and an overstated status, which is exactly why each one survived being written, re-read, and reviewed.
3. Concrete failure: a per-accelerator efficiency of 73% was described as what a cluster "measures". What was measured is the `busbw` curve; the 73% is derived from it under an assumed collective algorithm that has never been confirmed against NCCL's own logs. Fixed by saying the measurements "work out to" 73% and naming the assumption in a footnote.
4. Concrete failure: "at least 70% of the per-interface spec" was written into the cloud-provider chapter as a contract threshold. It came from one derived data point on one cluster, nothing established it as a pass mark, and no measurement of what that interface actually delivers exists yet - `ib_write_bw` on EFA is still unrun. Removed entirely.
5. Concrete failure: a B300 inter-node phase was said to "cross in 8.05ms". That is a wire-rate division of published specs, for a system this benchmark has never been run on, written in the indicative as though it were an elapsed time. The parent section had it right - "at wire rate that shard would cross in ... = 16.1ms, against 22.05ms measured" - and the later sentence dropped the qualifier.
6. A spec-rate division gives a bound on time, never the time. Keep the `at wire rate ... would` construction; as soon as it becomes `crosses in`, a reader takes it as an observation.
7. Carrying a figure from one system to another needs both the fact and the assumption stated. B300 sharing NVLink 5 with B200 does not make their intra-node `all-reduce` times equal - different HBM, different clocks - so reusing B200's 10.15ms was a second unstated assumption hidden inside the first.
8. Prefer saying a number is unknown over supplying a plausible one. "This benchmark has not been run on a B300 cluster" is worth more to a reader than any figure that would need to be labelled unreliable.

## Agent-invented content

1. An assistant working on this book will sometimes produce something genuinely new - a derivation, a conversion, a diagnostic, a way of framing a trade-off - that appears in no source and nowhere earlier in the chapter. That is welcome and worth having. Novelty is not the problem; unreviewed novelty is.
2. So the rule is not "never invent". Forbidding it outright would throw away the useful half and stifle the thing that makes a fresh pass worth running at all. The rule is that anything invented must be surfaced to the user as an invention, named as such, and kept out of the source files until the user has said it makes sense.
3. `Grounded` means one of exactly two things: an authoritative external reference, or content that already appears earlier in the chapter or in a cross-linked file. Anything else is invented - however obvious it feels, and however cleanly it follows from the material around it.
4. Say which kind of invention it is when flagging it, because they fail differently and need different checks. A derivation can be verified on the spot with algebra. A recommendation, threshold, technique, or diagnostic cannot - it needs a source or a measurement, and that is the kind that has caused every problem here so far.
5. The failure mode is not a wrong statement, it is a plausible one. A "run one accelerator per node" measurement method was proposed on 2026-08-02 with no source and a real failure mode, and read as established practice until it was researched and replaced with `ib_write_bw`. A `70%` acceptance threshold was invented and written straight into a chapter as a contract number. Neither looked like a hallucination at the time - that is the whole difficulty.
6. Flag it even when it is almost certainly right. The cost of asking is one sentence; the cost of a confident invention reaching a published chapter is that a reader acts on it.
7. This is stricter than [Change approval](#change-approval), which governs edits the user already has in view. An invention the user did not ask for and cannot see coming has to be named before it is applied, not folded silently into an approved batch.

## The author's own voice

1. This book is written in the first person and much of its value is the author's field experience - `I have noticed some distributed file systems`, `at JeanZay HPC we had to ask for a special dedicated partition`, `it took awhile to realize they were wrong`. That voice is evidence. A reader weighs `I got a whole lot of invalid reports because of it` differently from `this can produce invalid reports`, and rightly so. Which is exactly why it is never available for an assistant to write.
2. Never invent a first-person claim. Do not write `I have seen`, `in my experience`, `this cost me`, `we found`, or any other statement about what the author did, saw, measured, or suffered, unless those words already exist in the repository and you are preserving or relocating them.
3. Concrete failure: on 2026-08-04 a `hint:` explaining two `find` gotchas was given the clause "both of which cost me correct numbers before I understood them". Nothing anywhere in the repository said that. It was manufactured by pattern-matching the register of a genuine anecdote a few paragraphs up - the `--unlink=1` note, where the author really does report getting invalid reports and taking a while to realize it. Copying that key produced text that reads more authoritative than anything an assistant is entitled to write, and put a false memory into the author's mouth in the author's own book.
4. The tell is that it feels like good style. A section written in strong first person invites continuation in the same key, and the more convincingly the voice is copied the less likely anyone is to question it. Treat the urge to add colour as the signal to stop.
5. Converting an existing impersonal claim into a personal one is the same error in smaller form. The `O_DIRECT` logging paragraph said `typically helps`; rewriting it as `I have seen ... ` supplied an eyewitness the original did not contain. When a general claim genuinely needs weakening, weaken it with `can` or `may` and say what the limitation is - do not anchor it in an experience nobody reported.
6. Write the mechanism instead. `without -print the pruned folders are printed too, and their sizes get added to a user's total` is checkable by anyone; `this cost me correct numbers` is checkable by no one. The mechanism is also more useful, because a reader can tell whether it applies to their case.
7. The same restraint covers editorial characterizations pitched in the author's register - calling a workflow `genuinely painful`, a write-up `a goodie`, a vendor practice `a technology deficiency`. Those are judgements the author is entitled to make and an assistant is not, however obviously true they seem. `genuinely painful` was written into the Hugging Face cache note on 2026-08-04 and is the same class of thing, milder.
8. This is a specialization of [Agent-invented content](#agent-invented-content), and stricter in one respect: an invented derivation can at least be checked with algebra and kept once verified, whereas an invented memory cannot be verified by anyone except the author and has no repair other than deletion.

## Positional cross-references

1. `above` and `below` are only safe when the thing referred to is visible from where the reader is standing - the command, output block, table, or list item in the same section. Then leave them alone; a link to the section the reader is already in is noise.
2. Once a reference crosses a heading, replace the bare `above`/`below` with a Markdown link to the target section. The reader who arrived by deep link, search, or a cross-chapter jump has no `above`.
3. This applies to a `####` referring back to material in its parent `###`. The subsection is a landing point in its own right - `network/README.md`'s `So what should you expect?` is linked directly from the cloud-provider chapter - so "model 3 above" was unfollowable for anyone who entered there.
4. Prefer naming the target in the link text over keeping the direction word: `model 3 in [Inter-node speed depends on intra-node speed](#...)` rather than `[model 3 above](#...)`. Direction words go stale when sections are reordered; names don't.
5. Do not link when the target is already linked in the same sentence or paragraph. A second link to the same anchor a few words later adds nothing.
6. Leave comparative `above`/`below` alone - `60W above its TDP`, `below about 1GiB`, `approaches 0 from above` are quantities, not positions. Skipping these is the bulk of the work when sweeping a chapter.
7. Anchor arithmetic: an em dash in a heading is dropped and the spaces around it each become a hyphen, so `part 1 — do you need` yields `part-1--do-you-need` with a double hyphen. Verify a generated anchor against the file's own existing links rather than trusting a hand-rolled slugifier.

## Internal links and anchors

1. An anchor needs a document to attach to. Link to `network/README.md#anchor`, never to `network#anchor` - the latter resolves to a directory, and a fragment on a directory has nothing to bind to.
2. Concrete failure: `[Real network throughput](../network#real-network-throughput)` in the cloud-provider chapter. The heading was right and the generated anchor was right; the link form was not. GitHub resolves a relative link from a blob page to `/blob/master/network`, the blob URL of a directory, which does not render the README - so the fragment is dropped.
3. The `README.md#anchor` form is the dominant convention regardless: 130 links already used it against the 60 fixed on 2026-08-02, and all 60 of those were broken.
4. A link to a directory with *no* fragment is fine and needs no change. Only the anchored ones break, so a sweep should target `](path#anchor)` where `path` resolves to a directory.
5. Never let a link checker treat a directory as its `README.md`. A checker that quietly appends it is validating a resolution step that neither GitHub nor any local renderer performs, and it will pass every link in the class above. This is exactly how 60 broken links survived two separate "0 broken" reports in one session - the checker was wrong, not the links.
6. When generating anchors to check them, model GitHub faithfully: strip backticks and inline links from the heading text, lowercase it, drop every character that is not a word character, space, or hyphen, then replace each remaining space with a hyphen. Repeated identical headings get `-1`, `-2` appended in document order. Do not collapse runs of spaces - see [Positional cross-references](#positional-cross-references) item 7 for the em-dash case where that matters.
7. Sweep the whole repository, not the file just edited. The root `README.md` held 5 of the 60, and the first pass missed 11 more because its pattern assumed every relative path begins with `./` or `../`, which `compute/accelerator#...`, `benchmarks#...` and `../#...` do not.
8. `make check-links-local` runs `linkchecker` over the built HTML and is the authoritative check, but it depends on `markdown_it`, which is not installed as of 2026-08-02. Until it is, a hand-rolled scan is the fallback and is only worth running if it obeys items 5 and 6.
9. A link can resolve perfectly and still be broken. After moving or rewriting any block that other places point at, check the referrers, not the thing you just edited. On 2026-08-04 a `nvbandwidth` build recipe was relocated out of `### NVLink-C2C` into `network/benchmarks/README.md`; the PCIe section 150 lines earlier still said "with `nvbandwidth` built as shown under [NVLink-C2C](#nvlink-c2c)". The anchor resolved, the heading existed, and the build instructions the sentence promised were gone. No link checker catches this, because nothing is technically wrong.
10. The check is mechanical, so run it every time rather than trusting recall: grep for every inbound reference to the thing you moved, and for each one confirm the target still contains what the sentence claims it contains. `grep -n "#the-anchor" -r .` finds the referrers; reading each one finds the lies.
11. This generalizes past links. The same failure shape - editing outward-facing content and verifying only the edit - showed up three separate times in one session for tool invocations: `./nvbandwidth`, `./build/all_reduce_perf` and `./p2pBandwidthLatencyTest` were each dropped into a chapter with no account of where the binary comes from, because the assistant had them built and working in its own shell. Whenever a command lands in a chapter, confirm a reader who arrived at that line cold can obtain and run it - the build is either on the page or one link away.

## Suggestions report

1. Put a large set of findings in a repository file rather than only in chat.
2. Group findings by severity.
3. Use one flat numerical sequence. A number such as `2` must be sufficient; do not require prefixes such as `HIGH-02`.
4. Make each suggestion independently actionable so suggestions can be applied in any order.
5. When a suggestion is applied, remove it from the report.
6. Renumber the remaining suggestions when a batch is removed, so that the highest number equals the count of open items and the size of the queue is visible at a glance. Do it as a deliberate pass, not incidentally: renumber the open items into a compact `1..N` sequence, record the renumbering date at the top of the file, note that numbers appearing in the historical sections are pre-renumbering identifiers, and update every external reference in the same pass - `todo.md` and the assistant's own memory notes both cite these numbers by value.
7. The cost of renumbering is that a number stops being a durable identifier, so `do 39` in an old transcript no longer resolves. That is accepted deliberately in exchange for the count being legible; it is why the renumbering date and the historical-numbers note are not optional.
8. Include a numerical correction plan ordered by practical priority.

Use the newest `build/consistency-review-*.md` file in the current repository unless the user names a different report.

## Update opportunities report

1. Maintain a separate file for opportunities to extend the book when new accelerators, networking cards, switches, specifications, or standards become available.
2. Do not mix update opportunities with correctness findings.
3. Use the same stable flat-number workflow: make each item independent, remove applied items, and never renumber the remaining items.
4. While researching any topic, add newly discovered, primary-source-supported update opportunities to this file.
5. Propose each update before editing source content.
6. Finish the correctness suggestions before beginning the update queue unless the user explicitly changes that order.

Use the newest `build/update-suggestions-*.md` file in the current repository unless the user names a different report.

## Review scope

1. Check content, logic, numerical arithmetic, units, technical correctness, and internal consistency.
2. Ignore ordinary Markdown whitespace-only differences because rendering already ignores them. Tables are an exception, as covered by item 3: there whitespace carries readability meaning even though rendering discards it.
3. Keep Markdown table source vertically aligned. In tables, whitespace matters for maintainer readability even when rendering would be unchanged.
4. Check internal/local links, files, and anchors.
5. Do not perform external-link availability or liveness checks except for newly added links as specified below.
6. Re-validate every `as of this writing` clause. The phrase means "true when this section was written" rather than "true when the book was last updated" - which future-proofs the sentence for the author but tells the reader only that the claim may be stale, with no way to tell how stale. There are 28 of them as of 2026-08-05, led by `network/README.md` with 7 and `compute/accelerator/README.md` with 5, then 3 each in `storage/README.md`, `inference/README.md` and `debug/pytorch.md`, and the remaining 7 spread one or two per file. Treat each as an item to check, never as a disclaimer that excuses staleness.
7. Grep for it case-insensitively. Roughly half the instances are sentence-initial `As of this writing`, so a case-sensitive grep silently reports about 60% of them - which is how this file came to claim 21 and name the wrong three files as the concentration. A count that comes out suspiciously tidy is worth re-running with `-i`.
8. When one checks out, prefer replacing the phrase with the actual date, since the book already does this elsewhere - `as of 2026-07-31`, `As of 2026-07-28`. A dated claim tells the next pass how old it is; `as of this writing` cannot. Use the date the claim was *verified*, not the date `git blame` reports, since blame returns last-touch and any mechanical sweep - a unit normalization, a launcher rename - resets it to today on lines whose claims are years old.
9. Claims about a *dependency's* capabilities or documentation rot fastest - much faster than hardware specs or arithmetic, which mostly stay put. Sweep for `you can't`, `there is no way to`, `not documented`, `not possible`, `they are working on`, and anything described as `new`. `new` rots fastest of all.
10. Concrete failure, both in one paragraph of `network/benchmarks/README.md` on 2026-08-04. It stated that you cannot know which algorithm NCCL selected, citing a December 2022 NCCL answer, while `network/README.md` had just been updated to instruct readers to read exactly that from `NCCL_DEBUG_SUBSYS=INIT,TUNING` - the book contradicted itself in two files. The same paragraph called `NVLS` "not documented anywhere" and said the inter-node version was still being worked on, when `NVLS` and `NVLSTree` had been documented `NCCL_ALGO` values since NCCL 2.17 and 2.18. Neither sentence was wrong when written; both had quietly stopped being true.
11. So when a chapter is edited because a dependency gained a feature, grep the neighbouring files for older statements about that same dependency. The new measurement is what reveals the old claim - nothing else will, because a stale "you can't" reads exactly like a true one.
12. The neighbour can be the next paragraph. In `network/README.md` the `Omni-Path` section says the product "comes with either 100 or 200Gbps bandwidth", concludes it is therefore unsuitable for ML, and is followed two lines later by the note that 400Gbps `CN5000` NICs shipped in June 2025. Both were added by the same author to the same section; the newer fact simply never prompted a re-read of the older conclusion. Sweeping a whole chapter for a claim's neighbours is not enough if you do not also read the sentences immediately around it.

## Outdated references and dead artifacts

1. Periodically sweep for content that has quietly aged out rather than become wrong. Two kinds: files nothing points at any more, and references to software or hardware so old that following them wastes the reader's time.
2. Find orphaned files by checking whether each non-`README` file is named anywhere else in the repository. On 2026-08-05 that turned up `network/benchmarks/results/disable-nvlink.md`, reachable only from its own directory index and therefore effectively unreachable while reading. `trash/` is untracked scratch and `stabs/incoming.md` is deliberately unlinked - its README says to ignore the directory - so exclude both rather than reporting them every pass.
3. Prefer inlining a result over parking it in a side file. The book is otherwise entirely inline, a side file breaks reading flow, and it sets a precedent for scattering. Both `results/` sub-directories were removed on 2026-08-05: `network/benchmarks/results/` where one file was orphaned and the other duplicated a chapter table, and `storage/benchmarks/results/` where the linked "summary" was byte-identical to the six tables printed directly beneath the link. In each case the only unique content was provenance - hardware, date, partition path - which belongs in the sentence that introduces the numbers.
4. Remember `chapters-md.txt` when a chapter file is added or removed, since it drives the build and the review's own scope count.
5. Sweep for aged technology references with a term list: `TITAN`, `K80`, `P100`, `V100`, `pytorch-1.`, `torch==1.`, `cuda-10`, `cuda-11`, `python2`, and old pinned library versions. Most hits will be legitimate - a generations table needs its earlier generations, and a dated bug report with a linked issue is exactly how such things should be recorded.
6. The ones to act on are those that send a reader somewhere useless. Concrete: `storage/README.md` explained a `fio-scan` failure as "you run it on a system with python2 installed by default", five years after Python 2 reached end of life, so the diagnosis pointed at a cause that no longer exists.
7. Check the artifact before deleting the advice. In that same case the script really does call bare `python` rather than `python3`, and many current systems ship only `python3` - so the failure mode is live and only its stated cause was obsolete. The note was trimmed to the current diagnosis rather than removed, which the sweep would have gotten wrong had the script not been read.

## Companion book sync

1. `ml-engineering/debug/pytorch.md` and `the-art-of-debugging/pytorch/README.md` are the same chapter maintained in two books. Any change to one must be ported to the other in the same pass, in whichever direction the edit started. This is not optional cleanup to schedule later - an unported edit is a divergence that only a cross-book comparison will ever surface, and nothing in either repository points at the other.
2. The same applies to the scripts the two books share. As of 2026-08-05 there are 15 and the expected state is byte-identical: `NicerTrace.py`, `printflock.py`, `see-mem-usage.py`, `underflow_overflow.py`, `torch-distributed-gpu-test.py`, and the `tiny-scripts`/dataset helpers. They sit at different paths - `debug/` and `*/tools/` in one book, `pytorch/`, `pytorch/code/` and `pytorch/tiny-scripts/` in the other - so compare by basename, not by path.
3. `README.md` is the one same-named file expected to differ, since each book has its own index. Do not report it.
4. Detect divergence by pairing same-named files by basename and byte-comparing. Excluding `.git/`, `build/` and `trash/`, every pair should be identical; a non-empty result is either an unported edit or a deliberate difference that belongs in the review's `Checks that passed` note.
5. Checking pitfall, 2026-08-05: a first attempt mapped `ml-engineering/debug/<file>` onto `the-art-of-debugging/<file>` and so compared almost nothing - it matched only the root `README.md`, reported that one expected difference, and looked like a clean pass. A cross-book check that finds nothing is more likely to be mis-wired than correct; confirm it is actually pairing the shared scripts before trusting a clean result.
6. Also exclude the sibling working copies in the parent directory - `ml-engineering1/`, `ml-engineering2/`, `the-art-of-debugging2/`. They are separate snapshots, they will match the pre-edit state and look like divergences, and they must not be edited.
7. Concrete failure, 2026-08-05. The `python -m torch.distributed.run` to `torchrun` sweep was applied across `ml-engineering` only. It broke `torch-distributed-gpu-test.py` out of byte-identity - a file the previous review had explicitly recorded as identical after an earlier port - and left the companion's copy of the chapter with 11 un-swept command sites plus one in its `SKILL.md`. Nothing in the `ml-engineering` tree could have caught it; only the cross-book compare did, a day later. The `-u` exception had to be preserved on both sides, so the port is a re-application of the rule and not a blind copy.

## Sources and citations

1. Prefer original and primary sources: vendor specifications, official documentation, standards, original papers, and upstream source repositories.
2. When support is missing, recommend a direct link to the relevant specification or primary documentation page.
3. External primary sources may be consulted for factual verification, but they must not be treated as an external-link liveness scan.
4. Always open and confirm the exact target of every new external link before adding it to the repository.
5. The new-link rule is a narrow exception to the no-external-liveness-sweep preference: do not broadly recheck pre-existing external links.
6. Where possible, place citations directly under a numerical table so readers can confirm the displayed values.
7. Keep citations close to the claims or rows they support.
8. Batch command-line liveness checks through `build/check-new-links.sh` and request one reusable approval for that wrapper rather than separate approval for each `curl` invocation.
9. When a vendor or standards-body page appears unreachable, do not conclude the source is unavailable and do not fall back to a secondary source. Many such sites sit behind bot mitigation that rejects a bare HTTP client while serving the same page to a browser user-agent, so a specialized fetch tool can report a false negative. Retry with a normal downloader and a browser user-agent - this is pre-authorized for verifying citations in this repository, and no further approval is needed:

```bash
wget --user-agent="Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:39.0) Gecko/20100101 Firefox/39.0" URL
```

10. A guessed URL that returns a page is not necessarily the page you wanted - check the title and body before quoting it. If a guess lands on a listing or tag page, follow the site's own links, or read its `/specifications`-style index, to reach the document itself.

## Cross-vendor hardware tables

1. Do not force unlike vendor specifications into an artificial normalized schema.
2. Preserve the vendor's documented reporting scope, such as per SM, per CU, per XCD, or per accelerator.
3. Do not derive per-accelerator totals from private local caches unless the derivation is explicitly useful, clearly labeled, and approved.
4. Prefer `not disclosed` or omission over an unsupported estimate.
5. Add a brief explanation of how to compare unlike cache or memory resources, including differences in scope, sharing, semantics, and performance.
6. For new cache-table additions, consider only high-end GPUs.
7. Research current primary specifications, including newly documented parts such as AMD Instinct MI455X, but propose rows before adding them.
8. When one table would mix broadly shared cache capacity with private local resources, prefer separate comparison and vendor-native tables.
9. Hardware whose specifications are published but which a reader cannot yet buy does not belong in the sorted body of a comparison table, because the declared sort would rank it above shipping hardware and imply it is the thing to reach for. Instead, close the sorted rows, add one completely empty row as a visual break, and list the upcoming entries below it:

```
| AWS EFA v1 (P4d)             |     4 |     100 |     50 |
|                              |       |         |        |
| Omni-Path CN6000 example     |     8 |     800 |    800 |
| InfiniBand GDR3200           |     2 |    1600 |    400 |
```

10. This keeps a genuinely useful signal - the reader sees what is coming next - while making it unmistakable that those rows aren't available yet. Sort the below-the-break rows among themselves by the table's declared order. Move a row up into the body only once the product ships, and delete the break when nothing is left below it.
11. State the availability basis rather than leaving it implied: add a `GA` column - generally available, i.e. a reader can actually buy or rent it - with values `Y`, `N`, or `?` where the vendor names a product but publishes no availability. The column name is deliberately two characters because the values are one; `Shipped` wasted width. Combine it with the break from item 9: `N` rows go below the break, `Y` rows stay in the sorted body, and `?` rows stay in the body with a note saying why they are unresolved.
12. `GA` belongs in a table when at least one row is not `Y`, since a uniformly `Y` column only adds width. Place it immediately before any trailing `Notes` or `Ref.` column so the reference stays last. Put a dated note next to any `?`.
13. "Published spec" is not "available", and the two diverge in more ways than one: a released standard can precede silicon by years (PCIe 6.0 hardware arrived about three years after its specification), a vendor can publish full specs while marking them "Preliminary information ... subject to change" (NVIDIA Rubin), and a part can ship on one bus generation while the same node already uses a newer one elsewhere (accelerators on PCIe Gen5 x16 while ConnectX-8 NICs use Gen6). Decide `GA` on whether it can be obtained, not on whether numbers exist.

## Product sync map

A product's name, specs and availability are spread across many sections, so a single-place edit leaves the book arguing with itself. The failure is not hypothetical: the accelerator tables were given `GA` columns marking GB200/GB300 as shipping while the chapter's own opening summary still called them "expected", and one list said `MI400X` while every table said `MI455X`. When adding a product, renaming one, or changing an availability status, walk the list below and fix or consciously skip each entry.

`compute/accelerator/README.md`:

1. `## Bird's eye view on the high end accelerator reality` - the per-category lists (GPUs, HPU, TPU, On Pods and racks). This is the section most likely to be forgotten because it is prose, it sits far above the tables, and it is the first thing a reader sees. It also ends with a `That's about it as of DATE` stamp that must be re-dated whenever the lists change.
2. `## Glossary` - any new abbreviation, in alphabetical position.
3. `#### TFLOPS comparison table` - dtype columns, `GA`, the availability break, and the numbered row notes below it.
4. `#### Maximum Achievable Matmul FLOPS comparison table` - measured, so a new product appears here only once someone has run the benchmark. Never carry a spec number into it.
5. `### Accelerator memory size and speed` - capacity, type, bandwidth, `GA`, break.
6. `### Caches` - both the cross-vendor comparison and the vendor-native tables, plus their sources.
7. `### Clock speed`, `### Power consumption` - each with its own `GA`, break, `Notes` numbering, and `as of DATE` line explaining `not disclosed` / `N/A`.
8. `### Cloud accelerators` - the per-vendor roadmap lists (NVIDIA, AMD, Intel, Amazon, Google, SambaNova). Status verbs live here, so this is where "supposed to become available mid-2024" rots.
9. `## Accelerators in detail` - the per-vendor deep dives (`### NVIDIA`, `### AMD`, `### Intel Gaudi`, `### AWS Trainium`, `### Google TPU`, `### Huawei Ascend`, `### Cerebras`, `### SambaNova`) and the matching `## API` subsections.

`network/README.md`:

1. `## Glossary and concepts` - new fabric or adapter abbreviations.
2. `### All-to-all bandwidth`, `### Peer-to-peer bandwidth` - the intra-node node-level tables; both carry `GA` and a break.
3. The scale-up fabric sections a new accelerator lands in: `### PCIe`, `### NVLink`, `### NVLink-C2C`, `### NVSwitch`, `### Infinity Fabric / xGMI`, `### NeuronLink v3`, `### UB Link (UnifiedBus)`, `### Ultra Accelerator Link (UALink)`.
4. `## Inter-node networking` - the node table, then `### Network adapters`, `### InfiniBand`, `### Switch platforms` and its `####` children, `### Reaching beyond the rack`.
5. The per-vendor scale-out sections: `### EFA`, `### Gaudi2 (inter-node)`, `### Gaudi3 (inter-node)`, `### HPE Slingshot interconnect`, `### GPUDirect-TCPX`, `### Omni-Path`.

Cross-chapter:

1. An accelerator's scale-up bandwidth appears in both books' chapters - the accelerator chapter quotes it in prose while the network chapter tables it. Change both.
2. `## Bird's eye view on the high end accelerator reality` in the accelerator chapter and `### Backend networking` in the network chapter both quote current typical top speeds. Neither cites the other, so both drift.
3. Grep the whole repo for the old product name before concluding a rename is done - `MI450` survived in exactly one list after every table had moved to `MI455X`.
4. After any of this, run `make fix-tables`, re-check internal anchors, and confirm no table sorts a not-yet-purchasable part above shipping hardware.

## Table ordering and source layout

1. Keep Markdown table source vertically aligned for maintainer readability.
2. Sort table rows by an explicit column and direction.
3. State immediately before the table which column controls the ordering and whether the order is ascending or descending.
4. When adding a row, insert it into the declared order rather than appending it arbitrarily.
5. If a table has no declared sort order, ask the user which order to use and propose an appropriate column before editing it.
6. Do not choose a numerical sort that implies comparability between semantically different vendor specifications.
7. Where practical, use compact source references in the table and place the full live-checked links immediately below it.
8. When a column header is much wider than its body cells, compact it with `<br>` **inside the single header row**, e.g. `| Platform/<br>example<br>node |`. Never spread a header over several pipe-delimited lines: GFM requires the delimiter row to be the second line of the table, so a multi-line header stops the table from being recognized and GitHub renders the header as literal `|` text with the continuation segments orphaned beneath it.
9. Source column width is set by the longest of the full header string and the body cells. Rendered width is set instead by the longest `<br>`-separated segment, which is why `<br>` narrows the table for the reader even though it lengthens the source line. Optimize for the rendered width; a long source header line is not a problem.
10. Prefer concise, unambiguous abbreviations such as `Uni-dir.` when a full term makes a compact table column unnecessarily wide.
11. Each table has an independent `Ref.` namespace starting at `1`. Ref columns are left-aligned because reference IDs are categorical and may contain multiple comma-separated values. Ref cells and source numbers are plain numbers without brackets or links; the source descriptions below the table contain the actual links.
12. Move explanatory qualifiers out of compact table headers and into nearby prose or notes when the qualifier does not distinguish the displayed values.
13. Format each per-table `Sources:` block as an explicit numbered list whose item numbers match that table's `Ref.` values; do not combine multiple sources into one paragraph.
14. After every table edit, shrink each source column to the minimum width required by its longest header or body cell, while preserving vertical pipe alignment.
15. Keep rendered tables compact to minimize line wrapping on narrow media. Compact disproportionately wide headers with `<br>` within the single header row, use concise labels, and move nonessential detail below the table without sacrificing clarity.
16. After editing a table, run `make fix-tables`. It joins multi-line headers into one row, inserts a missing blank line before a table, and re-pads misaligned pipes, then cross-checks the source table count against what `pandoc` renders. It reports `file:line` for everything it fixed, and flags what it cannot fix - such as ragged cell counts, where there is no way to know which cell is missing.

## Glossary sections

1. Keep each glossary list alphabetically sorted, case-insensitively, so `RoCE` sorts next to `RoE` and `xGMI` lands with the letters rather than after them. Insert a new entry in place; never append to the end.
2. Sort per list, not across the section. A chapter may hold several lists - for example an abbreviation list and a `Speed-related:` list - and each is sorted independently.
3. A list whose order is deliberately pedagogical rather than alphabetical, such as one introducing `Unidirectional` before `Bi-directional`, may keep that order. Say so in a note, otherwise the next pass will "fix" it.
4. When adding an abbreviation to a chapter, add it to that chapter's glossary in the same edit. This applies to anything a reader can't expand on sight - a `GA` table column, `busbw`, `SuperNIC` - and not to vendor names, product model numbers, or terms as widely known as `GPU` or `CPU`.
5. Periodically check both directions: abbreviations used in the body but missing from the glossary, and glossary entries no longer used anywhere in the chapter. Neither is fatal, but the first hurts readers and the second is dead weight.

## Source line layout

1. Keep each prose paragraph on one physical source line; do not wrap prose to a fixed line width.
2. Keep each Markdown list item on one physical source line unless it contains nested block content.
3. Preserve intentional blank lines between Markdown blocks.
4. Only code is subject to a line-width limit, which is 119 characters.
5. Wrap code according to the syntax and semantics of its language rather than applying prose-style reflow.

## Unit formatting

1. Write a value tight against its unit, with no separating space: `340Gbps`, `80GiB`, `125TFLOPS`, `700W`. This is the dominant convention across the book.
2. Prefer the `p`-suffixed spelling over the slash spelling: use `TBps` rather than `TB/s`.
3. When quoting a vendor or other external source verbatim, leave the original spelling untouched. The tight rule governs the book's own prose, not quoted material.
4. Unit spacing is a further exception to `Review scope` item 2, and a different one from tables: here the whitespace changes the rendered text, not just the source.
5. Re-check this periodically across the whole book and fix any drift. Spaced forms reappear as new material is added, and `TFLOPS` was historically spaced in most chapters, so it drifts first.
6. When fixing units inside a script, change the code and any captured output in the same pass so the two continue to agree. Prose chapters may be fixed independently of scripts.
7. After changing a unit inside a table, restore vertical pipe alignment and re-shrink the affected columns as required by `Table ordering and source layout`.
