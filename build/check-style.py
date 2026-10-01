#!/usr/bin/env python3
"""Report hard-wrapped prose paragraphs. The book writes one line per paragraph.

Wrapping is invisible when reading rendered Markdown and costly in the source: every later edit to
a wrapped paragraph produces a reflow diff instead of a content diff, so real changes become hard
to review. Three sections written on 2026-08-07 were wrapped at about 95 characters while the
surrounding file had a fifth of its prose lines over 200; unwrapping them cut 130 lines to 69, 71
to 38 and 26 to 10.

This exists because prose rules did not hold. Every SESSION.md rule backed by a script has held;
the wrapping convention lived only as an unwritten habit and was broken immediately by new prose.
Edits made *into* existing paragraphs inherit the line they are spliced into, so this only bites
on newly written text - exactly when the surrounding convention is easiest to forget.

Scope is deliberately one check. Related defects are already covered or were tried and dropped:

- **table alignment** - `build/fix-tables.py` already re-pads columns and asserts cell contents are
  unchanged, and its `--dry-run` exits non-zero, so it is already a CI check. Use that; do not
  hand-pad tables, which is how a table in `training/performance/README.md` went ragged across four
  successive edits.
- **bare acronyms** - tried and dropped. Flagging any acronym lacking a gloss in the same file gave
  458 hits across 47 files, because `NCCL`, `CUDA`, `HBM`, `SLURM` and `TFLOPS` are this book's
  ordinary vocabulary. Restricting it to acronyms glossed nowhere in the book misses the real cases,
  since `FMA`, `CTA` and `MIG` were each glossed in *another* chapter and bare where used. Telling
  "vocabulary the reader has" from "jargon needing a gloss" is a judgement about audience.
- **non-ASCII maths symbols** - tried and dropped. 47 of 54 hits were `x` in one chapter's ratio
  tables, which is that chapter's consistent style.

A check that fires 500 times gets ignored, and takes the useful checks with it.

Reports, never rewrites: an indented line may be a list continuation where the indent carries
meaning, so the fix needs judgement.

usage: python build/check-style.py [file ...]     (defaults to chapters-md.txt)
"""
import os, re, sys

FENCE = re.compile(r'^(```|~~~)')
LIST = re.compile(r'^\s*([-*+]\s|\d+[.)]\s)')
# not flowing prose: table, heading, quote, html, image, footnote, or indented
NOT_PROSE = re.compile(r'^(\||#|>|<|!\[|\[!\[|footnote:|\s)')
# a link row - bare brackets and one-link-per-line entries - where the line break is the layout
LINK_ROW = re.compile(r'^([\[\]]|\[[^\]]+\]\([^)]+\)\s*\|?)$')

def wrapped_paragraphs(lines):
    """Runs of more than one flowing-prose line.

    Unindented lines below a list item are lazy continuations and keep their own line - joining
    those flattened a one-link-per-line list in `resources/README.md` when this was first written.
    A blank line ends list context.

    A link row is exempt for the same reason: the companion book's `methodology/README.md` puts
    cheatsheet links one per line inside bare `[` `]` lines, so joining them would destroy the layout
    rather than unwrap a paragraph. Matching the lines themselves rather than tracking bracket state
    keeps an unclosed `[` from swallowing the rest of the file.
    """
    out, buf, start, fence, in_list = [], 0, 0, False, False
    for n, l in enumerate(lines, 1):
        if FENCE.match(l):
            if buf > 1: out.append((start, buf))
            buf, fence = 0, not fence
            continue
        if fence:
            continue
        if l.strip() == '' or LIST.match(l) or in_list or NOT_PROSE.match(l) or LINK_ROW.match(l):
            if l.strip() == '': in_list = False
            if LIST.match(l): in_list = True
            if buf > 1: out.append((start, buf))
            buf = 0
            continue
        if not buf: start = n
        buf += 1
    if buf > 1: out.append((start, buf))
    return out

files = sys.argv[1:] or [l.strip() for l in open('chapters-md.txt') if l.strip()]
total = 0
for f in files:
    if not os.path.isfile(f):
        print(f'MISSING {f}'); total += 1; continue
    for start, count in wrapped_paragraphs(open(f, encoding='utf-8').read().split('\n')):
        print(f'{f}:{start}: WRAPPED PROSE - {count} lines, should be 1')
        total += 1

print(f'\nchecked {len(files)} files: {total} problem(s)')
sys.exit(1 if total else 0)
