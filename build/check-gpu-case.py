#!/usr/bin/env python3
"""Report 'gpu' and 'gpus' written in lowercase in prose. The book writes 'GPU' and 'GPUs'.

Lowercase is right wherever the text is something a reader types or a program prints, so these are
not reported:

- fenced code blocks, inline code spans, <code>/<pre>/<kbd> elements, HTML tags and comments
- link targets `](...)`, reference definitions `[id]: url`, autolinks `<...>` and bare URLs - the
  anchor `#multi-gpu-...` of a heading that says "multi-GPU" is lowercase on GitHub regardless
- a word that is part of a flag, a setting, a path, a file name or an identifier: the
  whitespace-delimited token holding it starts with `-`, or contains `/ _ = @ $ % ~ #`, an inner
  `:` or an inner `.` - `--gpus`, `gres=gpu:8`, `gpu_id`, `torch-distributed-gpu-test.py`
- `gpu0`, `node0.gpu0` and other words with letters or digits attached

Words joined by a slash meaning "or", as in "ranks/gpus", are prose and are reported.

usage: python build/check-gpu-case.py [file ...]     (defaults to chapters-md.txt)
"""
import os, re, sys

WORD = re.compile(r'(?<![A-Za-z0-9_])([Gg][Pp][Uu])(s?)(?![A-Za-z0-9_])')
FENCE = re.compile(r'^\s*(`{3,}|~{3,})')
REFDEF = re.compile(r'^\s{0,3}\[[^\]]+\]:\s')
PROTECTED = [
    re.compile(r'(`+)(.+?)\1'),                                       # inline code
    re.compile(r'<(code|pre|kbd)\b[^>]*>.*?</\1>', re.I),             # inline code as HTML
    re.compile(r'</?[A-Za-z][^>]*>|<(?:https?|ftp|mailto):[^>]*>'),   # HTML tags and autolinks
    re.compile(r'(?:https?|ftp)://[^\s)>\]]+|www\.[^\s)>\]]+'),        # bare URLs
]
TOKEN_CHARS = re.compile(r'[A-Za-z0-9_\-./:=@$%~#+]')
ALTERNATION = re.compile(r'^[A-Za-z]+(?:-[A-Za-z]+)*(?:/[A-Za-z]+(?:-[A-Za-z]+)*)+$')
NOT_PROSE = re.compile(r'^-|[/_=@$%~#:.]')


def protected(line, in_comment):
    """A per-character mask of what is not prose, and whether an HTML comment is still open."""
    mask = bytearray(len(line))
    def cover(a, b):
        mask[a:b] = b'\x01' * (b - a)
    if REFDEF.match(line):
        cover(line.index(']:') + 2, len(line))
    for rx in PROTECTED:
        for m in rx.finditer(line):
            cover(m.start(), m.end())
    # link targets, balancing nested parens as in `Hopper_(microarchitecture)`
    i = line.find('](')
    while i >= 0:
        j, depth = i + 2, 1
        while j < len(line) and depth:
            depth += {'(': 1, ')': -1}.get(line[j], 0)
            j += 1
        cover(i + 2, j)
        i = line.find('](', j)
    # HTML comments, which may span lines
    i = 0
    while i < len(line):
        if in_comment:
            j = line.find('-->', i)
            end = len(line) if j < 0 else j + 3
            cover(i, end)
            if j < 0:
                break
            in_comment, i = False, end
        else:
            j = line.find('<!--', i)
            if j < 0:
                break
            in_comment, i = True, j
    return mask, in_comment


def token_at(line, s, e):
    while s > 0 and TOKEN_CHARS.match(line[s - 1]):
        s -= 1
    while e < len(line) and TOKEN_CHARS.match(line[e]):
        e += 1
    return line[s:e].rstrip('.,:;!?\'"')


def lowercase_gpu(lines):
    """(line number, word, replacement) for every 'gpu'/'gpus' in prose not written 'GPU'/'GPUs'."""
    fence, in_comment = None, False
    for n, line in enumerate(lines, 1):
        m = FENCE.match(line)
        if fence:
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence) \
                    and not line.strip()[len(m.group(1)):].strip():
                fence = None
            continue
        if m:
            fence = m.group(1)
            continue
        mask, in_comment = protected(line, in_comment)
        for w in WORD.finditer(line):
            want = 'GPU' + w.group(2)
            if w.group(0) == want or mask[w.start()]:
                continue
            tok = token_at(line, w.start(), w.end())
            if NOT_PROSE.search(tok) and not ALTERNATION.match(tok):
                continue
            yield n, w.group(0), want
    if fence:
        yield len(lines), 'UNCLOSED FENCE', None


files = sys.argv[1:] or [l.strip() for l in open('chapters-md.txt') if l.strip()]
total = 0
for f in files:
    if not os.path.isfile(f):
        print(f'MISSING {f}'); total += 1; continue
    for n, word, want in lowercase_gpu(open(f, encoding='utf-8').read().split('\n')):
        if want is None:
            print(f'{f}:{n}: UNCLOSED CODE FENCE - the rest of the file was not checked')
        else:
            print(f"{f}:{n}: LOWERCASE GPU - '{word}' in prose, write '{want}'")
        total += 1

print(f'\nchecked {len(files)} files: {total} problem(s)')
sys.exit(1 if total else 0)
