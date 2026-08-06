#!/usr/bin/env python3
"""Report external links whose URL has moved, so the book can cite the endpoint directly.

A redirect is invisible to a reader and to build/check-links.py - the old URL still works,
so nothing looks broken - yet it means the book names a location the project has left.
GitHub org renames are the common case: outlines-dev/outlines, TimDettmers/bitsandbytes.

Only cross-host and path changes are reported. An http->https upgrade, a gained or lost
trailing slash, and a dropped #fragment are normal and are not moves.

Needs network, so this is not part of the fast local pass - run it deliberately.

usage: python build/check-redirects.py [--jobs N] [file ...]   (defaults to chapters-md.txt)
"""
import re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

URL = re.compile(r'https?://[^)"\'>\s]+')
TRAILING = '.,;:!?`'

def extract(line):
    """URLs on one line, with two Markdown quirks handled.

    A closing paren ends the match, so `Hopper_(microarchitecture)` is truncated and then
    404s - re-balance it. And a URL wrapped in backticks picks the backtick up as its tail.
    """
    out = []
    for u in URL.findall(line):
        u = u.rstrip(TRAILING)
        if u.count('(') > u.count(')') and line[line.find(u) + len(u):].startswith(')'):
            u += ')'
        out.append(u)
    return out

# Hosts that redirect to signed, expiring, or geo-specific endpoints. The final URL is not a
# canonical location and must never be pasted into the book - it rots within hours and pins
# the reader to one region's CDN.
CDN_HOSTS = ('cdn.hf.co', 'cloudfront.net', 'akamaized.net', 'blob.core.windows.net')

def normalize(u):
    """Strip the differences that are not moves, so only real relocations remain."""
    u = re.sub(r'^http://', 'https://', u)
    u = u.split('#')[0]
    # A clone URL's `.git` suffix is correct even though GitHub's web view redirects without it.
    u = re.sub(r'^(https://github\.com/[^/]+/[^/]+)\.git$', r'\1', u)
    return u.rstrip('/')

def final_url(url):
    """Where the URL actually lands, or None when it cannot be determined."""
    try:
        r = subprocess.run(
            ['curl', '-sIL', '--max-time', '20', '-o', '/dev/null',
             '-w', '%{url_effective}\t%{http_code}', url],
            capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return None, 'timeout'
    out = r.stdout.strip().split('\t')
    return (out[0], out[1]) if len(out) == 2 else (None, 'error')

args = [a for a in sys.argv[1:] if not a.startswith('--')]
jobs = 8
for a in sys.argv[1:]:
    if a.startswith('--jobs'):
        jobs = int(a.split('=')[1]) if '=' in a else 8

files = args or [l.strip() for l in open('chapters-md.txt') if l.strip()]

sites = {}                                             # url -> [(file, line), ...]
for f in files:
    try:
        lines = open(f, encoding='utf-8').read().split('\n')
    except OSError:
        print(f'MISSING CHAPTER {f}')
        continue
    for ln, line in enumerate(lines, 1):
        for u in extract(line):
            sites.setdefault(u, []).append((f, ln))

print(f'checking {len(sites)} distinct external URLs from {len(files)} files '
      f'with {jobs} parallel requests...\n')

moved, unreachable, cdn, dead = [], [], [], []
with ThreadPoolExecutor(max_workers=jobs) as pool:
    for url, (final, code) in zip(sites, pool.map(lambda u: final_url(u), sites)):
        if final is None or code in ('000', 'error', 'timeout'):
            unreachable.append((url, code))
            continue
        if any(h in final for h in CDN_HOSTS):
            cdn.append(url)                            # a download endpoint, not a new home
            continue
        if code == '404' or code.startswith('5'):
            dead.append((url, final, code))            # gone, not moved - needs a new source
            continue
        if normalize(final) != normalize(url):
            moved.append((url, final, code))

def suggested(url, final):
    """curl never sends the #fragment, so `final` always lacks it. Carry it over, or a naive
    replace silently downgrades a deep link to its page - and can collide with a sibling
    fragment on the same page, mangling the URL outright."""
    if '#' not in url:
        return final, ''
    frag = url.split('#', 1)[1]
    return f'{final.split("#")[0]}#{frag}', '  (fragment carried over - verify it still exists on the new page)'

for url, final, code in sorted(moved):
    where = sites[url][0]
    target, note = suggested(url, final)
    print(f'MOVED  {where[0]}:{where[1]}')
    print(f'  from {url}')
    print(f'  to   {target}   [{code}]{note}')
    if len(sites[url]) > 1:
        print(f'  ({len(sites[url])} occurrences)')

# A URL that is a strict prefix of another cannot be string-replaced first: doing so eats the
# separator or the fragment of its longer sibling. Replace longest-first.
prefixes = [(a, b) for a, _, _ in moved for b, _, _ in moved
            if a != b and b.startswith(a.rstrip('/'))]
if prefixes:
    print(f'\n{len(prefixes)} overlapping pair(s) - replace LONGEST FIRST or the shorter one '
          f'will corrupt the longer:')
    for a, b in sorted(set(prefixes)):
        print(f'  {a}\n    is a prefix of  {b}')

if dead:
    print(f'\n{len(dead)} URL(s) are GONE, not moved - these need a replacement source, '
          f'and a redirect target that 404s is no better than the original:')
    for url, final, code in sorted(dead):
        where = sites[url][0]
        print(f'  [{code}] {where[0]}:{where[1]}  {url}')
        if normalize(final) != normalize(url):
            print(f'         redirects to {final}, which also fails')

if cdn:
    print(f'\n{len(cdn)} URL(s) resolve to a signed or region-specific CDN endpoint. '
          f'These are working download links - do NOT replace them with where they land:')
    for url in sorted(cdn):
        print(f'  {url}')

if unreachable:
    print(f'\n{len(unreachable)} URL(s) could not be checked '
          f'(blocked, rate-limited, or offline) - not necessarily dead:')
    for url, code in sorted(unreachable):
        print(f'  [{code}] {url}')

print(f'\n{len(sites)} checked: {len(moved)} moved, {len(dead)} dead, {len(cdn)} CDN, '
      f'{len(unreachable)} unreachable')
sys.exit(1 if moved or dead else 0)
