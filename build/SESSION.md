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

## Suggestions report

1. Put a large set of findings in a repository file rather than only in chat.
2. Group findings by severity.
3. Use one flat numerical sequence. A number such as `2` must be sufficient; do not require prefixes such as `HIGH-02`.
4. Make each suggestion independently actionable so suggestions can be applied in any order.
5. When a suggestion is applied, remove it from the report.
6. Never renumber the remaining suggestions after removing an applied item.
7. Include a numerical correction plan ordered by practical priority.

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
