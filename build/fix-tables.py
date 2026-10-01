#!/usr/bin/env python

"""

Fix Markdown tables so they actually render, and report anything that can't be fixed automatically.

GitHub-Flavored Markdown is strict about two things that are easy to get wrong and impossible to
notice in the source:

1. the delimiter row (`| --- | --: |`) must be the *second* line of the table. A header spread over
   several pipe-delimited lines stops the table from being recognized at all - GitHub then shows the
   header as literal `|` text with the continuation segments orphaned beneath it. To stack a header
   over several rendered lines, `<br>` goes inside the single header row instead:
   `| Platform/<br>example<br>node |`

2. a table must be preceded by a blank line. Without it the table is absorbed into the preceding
   paragraph and every row renders as literal text.

Neither failure is visible when reading the Markdown source, which is why this exists.

Fixed automatically:

- a header spanning several rows is joined into one row, inserting `<br>` between segments
- a missing blank line before the table is inserted
- misaligned pipes are re-padded, shrinking each column to the width its longest cell needs

Reported but not fixed:

- ragged cell counts across rows, since there is no way to know which cell is missing or extra
- any table whose rendered count still disagrees with the source count after fixing

Cell contents are never altered: after re-padding, the script asserts that every cell is unchanged,
and after joining a header it asserts that all original header segments survive in the joined text.
Only whitespace and the `<br>` joins move.

Only the files listed in `chapters-md.txt` are processed - that is the book. Anything else in the
repository (`SKILL.md`, notes under `build/`, benchmark result dumps) is deliberately out of scope.
Pass paths explicitly to process files outside that list.

Usage:

    python build/fix-tables.py                       # fix the book
    python build/fix-tables.py network/README.md      # fix specific files
    python build/fix-tables.py --dry-run              # report only, change nothing
    python build/fix-tables.py --no-align             # don't touch pipe alignment

`--dry-run` exits non-zero if anything would change, so it doubles as a CI check.

If `pandoc` is installed, each file is rendered afterwards and the number of `<table>` elements is
compared against the number of tables in the source - a mismatch means a table still isn't
rendering for a reason this script doesn't model yet.

"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

# a delimiter cell is dashes with optional leading/trailing colons - one dash is enough for GFM
DELIM_CELL = re.compile(r"^:?-+:?$")

MARKER = {
    "left":   lambda w: ":" + "-" * (w - 1),
    "right":  lambda w: "-" * (w - 1) + ":",
    "center": lambda w: ":" + "-" * (w - 2) + ":",
    "none":   lambda w: "-" * w,
}


def split_cells(line):
    """Return the cells of a table row, without the outer pipes."""
    s = line.strip()
    return s[1:-1].split("|") if s.startswith("|") and s.endswith("|") else s.split("|")


def is_delimiter(line):
    cells = split_cells(line)
    return bool(cells) and all(DELIM_CELL.match(c.strip()) for c in cells)


def find_tables(lines):
    """Yield (start, end) for each run of table-ish lines that contains a delimiter row."""
    in_fence = False
    i = 0
    while i < len(lines):
        if lines[i].startswith("```"):
            in_fence = not in_fence
            i += 1
            continue
        if not in_fence and lines[i].lstrip().startswith("|"):
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            if any(is_delimiter(l) for l in lines[i:j]):
                yield i, j
            i = j
            continue
        i += 1


def column_alignments(delim_row):
    aligns = []
    for cell in split_cells(delim_row):
        cell = cell.strip()
        left, right = cell.startswith(":"), cell.endswith(":")
        aligns.append("center" if left and right else
                      "right" if right else
                      "left" if left else "none")
    return aligns


def join_header(header_rows, n_cols):
    """Collapse several header rows into one, joining each column's segments with <br>."""
    joined = []
    for col in range(n_cols):
        segments = [split_cells(r)[col].strip() for r in header_rows]
        segments = [s for s in segments if s]
        text = ""
        for k, segment in enumerate(segments):
            if k and not text.endswith("<br>"):
                text += "<br>"
            text += segment
        joined.append(text)
    return joined


def render_block(header, body, aligns):
    widths = [max([len(header[c])] + [len(r[c]) for r in body] + [4]) for c in range(len(aligns))]

    def cell(text, col, is_header):
        width = widths[col]
        if is_header or aligns[col] in ("left", "none"):
            return text.ljust(width)
        if aligns[col] == "right":
            return text.rjust(width)
        pad = (width - len(text)) // 2
        return " " * pad + text + " " * (width - len(text) - pad)

    out = ["| " + " | ".join(cell(header[c], c, True) for c in range(len(aligns))) + " |",
           "| " + " | ".join(MARKER[aligns[c]](widths[c]) for c in range(len(aligns))) + " |"]
    for row in body:
        out += ["| " + " | ".join(cell(row[c], c, False) for c in range(len(aligns))) + " |"]
    return out


def fix_table(block, fix_align=True):
    """Return (new_block, fixes, unfixable) for one table block."""
    fixes, unfixable = [], []
    delim = next(k for k, l in enumerate(block) if is_delimiter(l))
    n_cols = len(split_cells(block[delim]))

    counts = {len(split_cells(l)) for l in block}
    if len(counts) > 1:
        unfixable.append(f"ragged cell counts across rows {sorted(counts)} - "
                         f"cannot tell which cell is missing or extra")
        return block, fixes, unfixable

    aligns = column_alignments(block[delim])

    if delim > 1:
        header = join_header(block[:delim], n_cols)
        for original in block[:delim]:
            for col, segment in enumerate(split_cells(original)):
                segment = segment.strip()
                if segment and segment not in header[col]:
                    unfixable.append(f"header segment {segment!r} would be lost when joining")
                    return block, fixes, unfixable
        fixes.append(f"joined a {delim}-line header into one row using <br>")
    else:
        header = [c.strip() for c in split_cells(block[0])]

    body = [[c.strip() for c in split_cells(l)] for l in block[delim + 1:]]
    new = render_block(header, body, aligns)

    misaligned = len({tuple(k for k, c in enumerate(l) if c == "|") for l in block}) > 1
    if delim > 1 or (fix_align and misaligned):
        # never let contents change - only whitespace and the <br> joins may move
        before = [[c.strip() for c in split_cells(l)] for l in block[delim + 1:]]
        after = [[c.strip() for c in split_cells(l)] for l in new[2:]]
        assert before == after, "internal error: body cells changed"
        if delim == 1 and misaligned:
            fixes.append("re-padded misaligned pipes")
        return new, fixes, unfixable

    return block, fixes, unfixable


def process_file(path, fix_align=True, dry_run=False):
    lines = Path(path).read_text().split("\n")
    original = list(lines)
    reports = []

    # bottom-up so earlier line numbers stay valid as the file grows or shrinks
    for start, end in reversed(list(find_tables(lines))):
        block = lines[start:end]
        try:
            new, fixes, unfixable = fix_table(block, fix_align=fix_align)
        except StopIteration:
            continue

        needs_blank = start > 0 and lines[start - 1].strip()
        if needs_blank:
            fixes.append("inserted the missing blank line before the table")

        for message in unfixable:
            reports.append((start + 1, "NEEDS ATTENTION", message))
        for message in fixes:
            reports.append((start + 1, "fixed", message))

        if unfixable:
            continue
        lines[start:end] = new
        if needs_blank:
            lines.insert(start, "")

    changed = lines != original
    if changed and not dry_run:
        Path(path).write_text("\n".join(lines))
    return changed, sorted(reports)


def rendered_table_count(path):
    """Number of <table> elements pandoc produces, or None if pandoc is unavailable/failed."""
    if not shutil.which("pandoc"):
        return None
    result = subprocess.run(["pandoc", "-f", "gfm", "-t", "html", str(path)],
                            capture_output=True, text=True)
    return result.stdout.count("<table>") if result.returncode == 0 else None


def source_table_count(path):
    lines = Path(path).read_text().split("\n")
    return sum(1 for _ in find_tables(lines))


def main():
    parser = argparse.ArgumentParser(description="fix Markdown tables so they render")
    parser.add_argument("files", nargs="*",
                        help="files to process (default: the book, per chapters-md.txt)")
    parser.add_argument("--dry-run", action="store_true", help="report only, change nothing")
    parser.add_argument("--no-align", action="store_true", help="don't touch pipe alignment")
    args = parser.parse_args()

    if args.files:
        files = [Path(f) for f in args.files]
    else:
        listing = Path("chapters-md.txt")
        if not listing.exists():
            sys.exit("chapters-md.txt not found - run from the repository root, or pass files")
        files = [Path(l.strip()) for l in listing.read_text().split("\n") if l.strip()]

    n_fixed = n_attention = n_tables = n_files_changed = 0

    for path in files:
        if not path.exists():
            print(f"{path}: missing, skipped")
            continue
        n_tables += source_table_count(path)
        changed, reports = process_file(path, fix_align=not args.no_align, dry_run=args.dry_run)
        n_files_changed += bool(changed)

        for line_no, kind, message in reports:
            print(f"{path}:{line_no}: {kind}: {message}")
            if kind == "fixed":
                n_fixed += 1
            else:
                n_attention += 1

        rendered = rendered_table_count(path)
        if not args.dry_run and rendered is not None:
            source = source_table_count(path)
            if source and rendered != source:
                print(f"{path}: NEEDS ATTENTION: {source} tables in source but pandoc "
                      f"rendered {rendered}")
                n_attention += 1

    verb = "would fix" if args.dry_run else "fixed"
    print(f"\n{n_tables} tables in {len(files)} files: {verb} {n_fixed}, "
          f"{n_attention} need attention, {n_files_changed} files "
          f"{'would change' if args.dry_run else 'changed'}")
    if not shutil.which("pandoc"):
        print("note: pandoc not found - skipped the source-vs-rendered cross-check")

    if args.dry_run:
        return 1 if (n_fixed or n_attention) else 0
    return 1 if n_attention else 0


if __name__ == "__main__":
    sys.exit(main())
