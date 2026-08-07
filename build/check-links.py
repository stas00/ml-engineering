#!/usr/bin/env python3
"""Scan local Markdown links and anchors across the book.

Deliberately does NOT treat a directory as its README.md - GitHub does not do that for an
anchored link, and a checker that does will pass every link in that broken class. See
build/SESSION.md "Internal links and anchors".

usage: python build/check-links.py [file ...]     (defaults to chapters-md.txt)
"""
import os, re, sys

LINK = re.compile(r'!?\[[^\]]*\]\(\s*([^)\s]+?)\s*(?:"[^"]*")?\)')
CODE = re.compile(r'`[^`]*`')
SCHEME = re.compile(r'https?://')

# External URLs are otherwise out of scope here - build/check-redirects.py resolves those over the
# network. But one defect is detectable locally and cheaply, so it is caught on every run: a URL
# wrapped into another URL. An archive citation legitimately carries two schemes
# (web.archive.org/web/<ts>/https://original), so three or more means a replacement was applied to a
# URL that was a substring of a longer one. Concrete near-miss: swapping a withdrawn NVIDIA blog URL
# for its Wayback capture, when the same URL already appeared inside an existing archive link 423
# lines below, would have produced web.archive.org/.../web.archive.org/.../developer.nvidia.com.
# See SESSION.md "External link rot".

def slug(heading):
    """Model GitHub's anchor generation: strip inline links and backticks, lowercase,
    drop anything that is not word char / space / hyphen, spaces to hyphens."""
    h = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', heading)   # inline links -> text
    h = h.replace('`', '')
    h = h.lower()
    h = re.sub(r'[^\w\s-]', '', h)
    return re.sub(r'\s', '-', h.strip())

def anchors_of(path):
    """All anchors a file offers, with GitHub's -1/-2 suffixes for duplicates."""
    out, seen = set(), {}
    try:
        lines = open(path, encoding='utf-8').read().split('\n')
    except OSError:
        return out
    fenced = False
    for line in lines:
        if line.lstrip().startswith('```'):
            fenced = not fenced
            continue
        if fenced or not line.startswith('#'):
            continue
        m = re.match(r'(#{1,6})\s+(.*)', line)
        if not m:
            continue
        s = slug(m.group(2))
        n = seen.get(s, 0)
        seen[s] = n + 1
        out.add(s if n == 0 else f'{s}-{n}')
    return out

files = sys.argv[1:] or [l.strip() for l in open('chapters-md.txt') if l.strip()]
anchor_cache = {}
n_links = n_bad = 0

for f in files:
    if not os.path.isfile(f):
        print(f'MISSING CHAPTER {f}')
        n_bad += 1
        continue
    base = os.path.dirname(f)
    fenced = False
    for ln, line in enumerate(open(f, encoding='utf-8'), 1):
        if line.lstrip().startswith('```'):
            fenced = not fenced
            continue
        if fenced:                                         # not rendered as links
            continue
        for target in LINK.findall(CODE.sub('', line)):    # nor is `[x](y)` in backticks
            if target.startswith(('http://', 'https://', 'mailto:')):
                n_schemes = len(SCHEME.findall(target))
                if n_schemes >= 3:
                    print(f'{f}:{ln}: MALFORMED URL - scheme appears {n_schemes}x, so a URL is '
                          f'wrapped inside another -> {target}')
                    n_bad += 1
                continue
            n_links += 1
            path, _, anchor = target.partition('#')
            if not path:                                   # same-file anchor
                resolved = f
            else:
                resolved = os.path.normpath(os.path.join(base, path))
                if not os.path.exists(resolved):
                    print(f'{f}:{ln}: MISSING PATH -> {target}')
                    n_bad += 1
                    continue
                if os.path.isdir(resolved):
                    if anchor:                             # the broken class
                        print(f'{f}:{ln}: ANCHOR ON DIRECTORY -> {target}')
                        n_bad += 1
                    continue
            if anchor:
                if resolved not in anchor_cache:
                    anchor_cache[resolved] = anchors_of(resolved)
                if anchor.lower() not in anchor_cache[resolved]:
                    print(f'{f}:{ln}: MISSING ANCHOR -> {target}')
                    n_bad += 1

print(f'\nscanned {n_links} local links in {len(files)} files: {n_bad} problem(s)')
sys.exit(1 if n_bad else 0)
