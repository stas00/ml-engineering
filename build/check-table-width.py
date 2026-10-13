#!/usr/bin/env python3
"""Flag Markdown tables wider than the PDF text block.

Prince renders this book at US Letter with the stylesheet in ``prince_style.css``:
2cm page margins, a 10pt Gentium body, ``code`` at 0.8em, and tables at
``width: auto`` with a 1em side margin and 0.4em cell padding. A cell wraps at
spaces, so a long sentence does not make the column wide. What cannot wrap is
a single word. ``<br>`` is the same kind of break as a space: ``word<br>word``
is ``word word``. Headers use it so the column is only as wide as its longest
word, not as wide as the whole header.

Measured with Prince 16.1 on 2026-10-13. The text block is 498.6pt wide. The
table's side margins and the outer cell padding take 28pt of that, and each
extra column costs another 8.2pt of padding between the columns. What remains
for the glyphs is 480.5pt. A column's width is its longest unbreakable word
plus that padding. The words are what is left after splitting on whitespace
and on ``<br>``. Backticked text is the monospace, 4.82pt per character.
Everything else is Gentium at 10pt. Bold is the same width as regular. Link
targets and ``**`` do not render, so they are not counted. A word with no
break that is wider than the room left for its column runs off the page.

``--self-test`` checks the width model against those Prince measurements. It
does not need Prince.

usage: python build/check-table-width.py [file ...]   (default: chapters-md.txt)
"""

import importlib.util
import re
import sys
from pathlib import Path

# Glyphs must end early enough that the cell's right padding still lands inside
# the text block. Prince 16.1, letter, prince_style.css, 2026-10-13.
BUDGET_PT = 480.5
COL_GAP_PT = 8.2
CODE_PT = 4.82
# Wider than any measured letter, so an unmeasured glyph cannot hide an overflow.
UNKNOWN_PT = 8.50

# Advances in pt at 10pt Gentium. Space is the gap measured between consecutive
# "n" glyphs. The rest are one glyph per word from the same render, with the
# page number (which Prince draws in the footer) left out.
GLYPH = {
    " ": 2.20, "!": 2.71, '"': 4.24, "#": 4.69, "$": 4.69, "%": 6.89, "&": 6.66,
    "'": 2.29, "(": 3.16, ")": 3.16, "*": 4.46, "+": 4.10, ",": 2.29, "-": 3.36,
    ".": 2.29, "/": 4.69, "0": 4.69, "1": 4.69, "2": 4.69, "3": 4.69, "4": 4.69,
    "5": 4.69, "6": 4.69, "7": 4.69, "8": 4.69, "9": 4.69, ":": 2.29, ";": 2.29,
    "<": 4.38, "=": 4.38, ">": 4.38, "?": 4.33, "@": 8.07, "A": 5.95, "B": 5.49,
    "C": 5.36, "D": 5.92, "E": 4.96, "F": 4.78, "G": 5.83, "H": 6.53, "I": 2.97,
    "J": 3.05, "K": 5.78, "L": 4.76, "M": 8.06, "N": 6.48, "O": 6.00, "P": 5.22,
    "Q": 6.00, "R": 5.61, "S": 4.76, "T": 5.50, "U": 6.53, "V": 6.49, "W": 8.50,
    "X": 6.15, "Y": 5.94, "Z": 5.22, "[": 3.08, "\\": 4.69, "]": 3.08, "^": 4.92,
    "_": 4.69, "`": 3.18, "a": 4.58, "b": 5.07, "c": 4.37, "d": 5.20, "e": 4.61,
    "f": 3.11, "g": 4.87, "h": 5.51, "i": 2.70, "j": 2.58, "k": 5.00, "l": 2.70,
    "m": 8.03, "n": 5.51, "o": 5.02, "p": 5.29, "q": 5.15, "r": 3.95, "s": 3.86,
    "t": 3.44, "u": 5.29, "v": 4.93, "w": 7.06, "x": 5.02, "y": 4.93, "z": 4.41,
    "{": 3.35, "|": 2.29, "}": 3.35, "~": 4.87,
    # measured the same way, because these occur in table cells
    "×": 4.10, "—": 7.66, "–": 4.69, "µ": 5.39, "−": 4.10, "±": 4.10,
}

ENTITIES = {"amp": "&", "lt": "<", "gt": ">", "quot": '"', "apos": "'", "nbsp": " "}
BR = re.compile(r"<br\s*/?>", re.I)
LINK = re.compile(r"!\[([^\]]*)\]\([^)]*\)|\[([^\]]*)\]\([^)]*\)")
TAG = re.compile(r"<[^>]+>")
ENTITY = re.compile(r"&(#\d+|#x[0-9a-fA-F]+|[a-z]+);")
CODE = re.compile(r"`([^`]*)`")


def load_fix_tables():
    path = Path(__file__).with_name("fix-tables.py")
    spec = importlib.util.spec_from_file_location("fix_tables", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def decode_entities(text):
    def sub(match):
        token = match.group(1)
        if token in ENTITIES:
            return ENTITIES[token]
        if token.startswith("#x"):
            return chr(int(token[2:], 16))
        if token.startswith("#"):
            return chr(int(token[1:]))
        return match.group(0)
    return ENTITY.sub(sub, text)


def plain_width(text, unknown):
    width = 0.0
    for ch in text:
        if ch in GLYPH:
            width += GLYPH[ch]
        else:
            unknown.add(ch)
            width += UNKNOWN_PT
    return width


def visible_source(part):
    """The text Prince draws for one ``<br>`` piece. ``<br>`` itself is already split off."""
    part = decode_entities(part)

    def link(match):
        # group 1 is an image alt, group 2 is a link's text. The target is not drawn.
        return match.group(1) if match.group(1) is not None else match.group(2)

    part = LINK.sub(link, part)
    part = part.replace("**", "").replace("__", "")
    part = TAG.sub("", part)
    return part


def segment_width(segment, unknown):
    """Width of one unbreakable word, after the markdown that does not draw."""
    width = 0.0
    pos = 0
    for match in CODE.finditer(segment):
        width += plain_width(segment[pos:match.start()], unknown)
        width += CODE_PT * len(match.group(1))
        pos = match.end()
    width += plain_width(segment[pos:], unknown)
    return width


def unbreakable_words(cell):
    """Words left after ``<br>`` and spaces. ``word<br>word`` is two words, like ``word word``."""
    words = []
    for part in BR.split(cell):
        words.extend(visible_source(part).split())
    return words


def table_width(block, fix_tables):
    """Return (width_pt, n_cols, longest_words, unknown_chars) or None if ragged.

    Width is the longest unbreakable word in each column, plus the cell padding
    between columns. ``<br>`` breaks a word the way a space does.
    """
    delim = next(k for k, line in enumerate(block) if fix_tables.is_delimiter(line))
    n_cols = len(fix_tables.split_cells(block[delim]))
    rows = [fix_tables.split_cells(line) for i, line in enumerate(block) if i != delim]
    if any(len(row) != n_cols for row in rows):
        return None
    unknown = set()
    columns = []
    for col in range(n_cols):
        widest = 0.0
        sample = ""
        for row in rows:
            for word in unbreakable_words(row[col]):
                width = segment_width(word, unknown)
                if width > widest:
                    widest = width
                    sample = word
        columns.append((widest, sample))
    gaps = COL_GAP_PT * (n_cols - 1)
    return sum(w for w, _ in columns) + gaps, n_cols, columns, unknown


def line_of(start):
    return start + 1


def check_files(files, fix_tables):
    failures = 0
    tables = 0
    widest = None
    for path in files:
        if not path.is_file():
            print(f"MISSING {path}")
            failures += 1
            continue
        lines = path.read_text(encoding="utf-8").split("\n")
        for start, end in fix_tables.find_tables(lines):
            tables += 1
            try:
                measured = table_width(lines[start:end], fix_tables)
            except StopIteration:
                continue
            if measured is None:
                print(f"{path}:{line_of(start)}: TABLE WIDTH skipped - ragged cell counts")
                failures += 1
                continue
            width, n_cols, columns, unknown = measured
            if widest is None or width > widest[0]:
                widest = (width, path, line_of(start))
            if width <= BUDGET_PT:
                continue
            failures += 1
            over = width - BUDGET_PT
            print(f"{path}:{line_of(start)}: TABLE WIDTH {width:.0f}pt, "
                  f"{over:.0f}pt over the {BUDGET_PT:.0f}pt PDF budget ({n_cols} columns)")
            for index, (col_width, sample) in enumerate(columns):
                if col_width < 40:
                    continue
                shown = sample if len(sample) <= 70 else sample[:67] + "..."
                print(f"    col {index + 1}: longest word {col_width:.0f}pt  {shown}")
            if unknown:
                chars = " ".join(sorted(unknown))
                print(f"    unmeasured glyphs (counted at {UNKNOWN_PT:.1f}pt): {chars}")
    widest_bit = ""
    if widest:
        widest_bit = f", widest {widest[0]:.0f}pt at {widest[1]}:{widest[2]}"
    print(f"\nchecked {len(files)} files, {tables} tables: "
          f"{failures} over the {BUDGET_PT:.0f}pt budget{widest_bit}")
    return failures


def self_test():
    """The Prince renders these cases were measured against. Widths, not pixels."""
    fix_tables = load_fix_tables()
    cases = [
        ("|" + "n" * 85 + "|\n|---|\n|x|", False),
        ("|" + "n" * 90 + "|\n|---|\n|x|", True),
        ("|" + "M" * 50 + "|\n|---|\n|x|", False),
        ("|" + "M" * 60 + "|\n|---|\n|x|", True),
        # plain "m" is 8.03pt wide; the same letters in `code` are 4.82pt
        ("|" + "m" * 40 + "|\n|---|\n|x|", False),
        ("|" + "`" + "m" * 90 + "`" + "|\n|---|\n|x|", False),
        ("|" + "`" + "m" * 100 + "`" + "|\n|---|\n|x|", True),
        ("|" + "|".join(["n" * 28] * 3) + "|\n|" + "|".join(["---"] * 3) + "|\n|"
         + "|".join(["x"] * 3) + "|", False),
        ("|" + "|".join(["n" * 32] * 3) + "|\n|" + "|".join(["---"] * 3) + "|\n|"
         + "|".join(["x"] * 3) + "|", True),
        ("|" + "|".join(["n" * 16] * 5) + "|\n|" + "|".join(["---"] * 5) + "|\n|"
         + "|".join(["x"] * 5) + "|", False),
        ("|" + "|".join(["n" * 18] * 5) + "|\n|" + "|".join(["---"] * 5) + "|\n|"
         + "|".join(["x"] * 5) + "|", True),
        ("|" + "|".join(["n" * 10] * 7) + "|\n|" + "|".join(["---"] * 7) + "|\n|"
         + "|".join(["x"] * 7) + "|", False),
        ("|" + "|".join(["n" * 12] * 7) + "|\n|" + "|".join(["---"] * 7) + "|\n|"
         + "|".join(["x"] * 7) + "|", True),
        # a 90-n body word still overflows; the header's <br> does not glue those n's together
        ("|n" + "<br>n" * 2 + "|\n|---|\n|" + "n" * 90 + "|", True),
        ("|" + "<br>".join(["n" * 30] * 3) + "|\n|---|\n|x|", False),
        # <br> is a space. 40 M's twice fits; glued into one word it would not
        ("|" + "M" * 40 + "<br>" + "M" * 40 + "|\n|---|\n|x|", False),
        ("|" + "M" * 40 + " " + "M" * 40 + "|\n|---|\n|x|", False),
        ("|" + "M" * 80 + "|\n|---|\n|x|", True),
        # a sentence wraps. only its longest word counts
        ("|" + "the quick brown fox " * 40 + "|\n|---|\n|x|", False),
        # the link target is not drawn
        ("|[" + "n" * 20 + "](https://example.com/" + "n" * 80 + ")|\n|---|\n|x|", False),
    ]
    failed = 0
    for src, expect_over in cases:
        block = [line for line in src.split("\n") if line]
        measured = table_width(block, fix_tables)
        over = measured[0] > BUDGET_PT
        if over != expect_over:
            failed += 1
            print(f"SELF-TEST FAIL width={measured[0]:.1f} over={over} expected={expect_over} {src[:60]!r}")
    # one real-shaped table: the fp4 header is short segments, so it fits
    fp4 = (
        "| Format | Block<br>size | Hardware |\n"
        "| --- | ---: | --- |\n"
        "| mxfp4 | 32 | NVIDIA Blackwell<br>AMD MI350X/MI355X<br>Huawei Ascend 950DT |\n"
    )
    measured = table_width([line for line in fp4.split("\n") if line], fix_tables)
    if measured[0] > BUDGET_PT:
        failed += 1
        print(f"SELF-TEST FAIL fp4-like table measured {measured[0]:.1f}pt")
    print(f"self-test: {failed} failure(s)")
    return failed


def main(argv):
    if "--self-test" in argv:
        return self_test()
    fix_tables = load_fix_tables()
    args = [a for a in argv if not a.startswith("-")]
    if args:
        files = [Path(a) for a in args]
    else:
        listing = Path("chapters-md.txt")
        if not listing.exists():
            sys.exit("chapters-md.txt not found - run from the repository root, or pass files")
        files = [Path(line.strip()) for line in listing.read_text().split("\n") if line.strip()]
    return 1 if check_files(files, fix_tables) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
