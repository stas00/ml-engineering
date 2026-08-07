#!/usr/bin/env python3
"""Report external links whose URL has moved or died, so the book can cite the endpoint.

A redirect is invisible to a reader and to build/check-links.py - the old URL still works,
so nothing looks broken - yet it means the book names a location the project has left.
GitHub org renames are the common case: outlines-dev/outlines, TimDettmers/bitsandbytes.

Needs network, so this is not part of the fast local pass - run it deliberately.

Politeness: requests to one domain are serialized with --delay seconds between them, while
different domains proceed in parallel. Never remove this. The book cites 243 distinct
github.com URLs and 46 on huggingface.co; firing those off concurrently looks like a scraper
and gets the runner throttled or IP-blocked, which is far more expensive than a slow check.
That floor makes a full run take roughly (largest per-domain count x delay) - about 12
minutes at the default. Raise --jobs to cover more domains at once, not to go faster on one.

usage: python build/check-redirects.py [--jobs N] [--delay SECS] [file ...]
       (defaults to chapters-md.txt)
"""
import re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

URL = re.compile(r'https?://[^)\]}"\'>\s]+')
TRAILING = '.,;:!?`'

# Some vendors reject a bare HTTP client outright - amd.com, hpe.com, microsoft.com, nasa.gov
# all do. Retrying with a browser user-agent separates real 404s from bot mitigation, which
# matters: amd.com's instinct/specifications.html looked merely "blocked" for a whole pass
# and was in fact gone. See SESSION.md "Sources and citations" item 9.
BROWSER_UA = ('Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:39.0) '
              'Gecko/20100101 Firefox/39.0')

# Hosts that redirect to signed, expiring, or geo-specific endpoints. The final URL is not a
# canonical location and must never be pasted into the book - it rots within hours and pins
# the reader to one region's CDN.
CDN_HOSTS = ('cdn.hf.co', 'cloudfront.net', 'akamaized.net', 'blob.core.windows.net')

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

def normalize(u):
    """Strip the differences that are not moves, so only real relocations remain."""
    u = re.sub(r'^http://', 'https://', u)
    u = u.split('#')[0]
    # A clone URL's `.git` suffix is correct even though GitHub's web view redirects without it.
    u = re.sub(r'^(https://github\.com/[^/]+/[^/]+)\.git$', r'\1', u)
    return u.rstrip('/')

def final_url(url, ua=None):
    """Where the URL actually lands, or None when it cannot be determined."""
    cmd = ['curl', '-sIL', '--max-time', '20', '-o', '/dev/null',
           '-w', '%{url_effective}\t%{http_code}']
    if ua:
        cmd += ['-A', ua]
    try:
        r = subprocess.run(cmd + [url], capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return None, 'timeout'
    out = r.stdout.strip().split('\t')
    return (out[0], out[1]) if len(out) == 2 else (None, 'error')

def resolve(url, delay):
    """A bare client first, then a browser user-agent if the host refused to answer at all."""
    final, code = final_url(url)
    if final is None or code in ('000', 'error', 'timeout'):
        time.sleep(delay)
        final, code = final_url(url, ua=BROWSER_UA)
    return final, code

def suggested(url, final):
    """curl never sends the #fragment, so `final` always lacks it. Carry it over, or a naive
    replace silently downgrades a deep link to its page - and can collide with a sibling
    fragment on the same page, mangling the URL outright."""
    if '#' not in url:
        return final, ''
    frag = url.split('#', 1)[1]
    return (f'{final.split("#")[0]}#{frag}',
            '  (fragment carried over - verify it still exists on the new page)')

args, jobs, delay = [], 8, 3.0
for a in sys.argv[1:]:
    if a.startswith('--jobs'):
        jobs = int(a.split('=')[1]) if '=' in a else jobs
    elif a.startswith('--delay'):
        delay = float(a.split('=')[1]) if '=' in a else delay
    elif not a.startswith('--'):
        args.append(a)

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

by_host = {}
for u in sites:
    by_host.setdefault(urlparse(u).hostname or '', []).append(u)

worst = max((len(v) for v in by_host.values()), default=0)
print(f'checking {len(sites)} distinct external URLs across {len(by_host)} domains from '
      f'{len(files)} files\n{jobs} domains at a time, {delay}s between requests to the same '
      f'domain - the busiest has {worst} URLs, so expect ~{int(worst * delay / 60)}+ min\n')

def check_host(host):
    """One domain's URLs, sequentially, spaced by `delay`. Called once per domain so that
    different domains overlap while a single domain is never hit concurrently."""
    out = []
    for i, u in enumerate(by_host[host]):
        if i:
            time.sleep(delay)
        out.append((u, *resolve(u, delay)))
    return out

results = []
with ThreadPoolExecutor(max_workers=jobs) as pool:
    for chunk in pool.map(check_host, by_host):
        results.extend(chunk)

moved, unreachable, cdn, dead = [], [], [], []
for url, final, code in results:
    if final is None or code in ('000', 'error', 'timeout'):
        unreachable.append((url, code))
    elif any(h in final for h in CDN_HOSTS):
        cdn.append(url)                                # a download endpoint, not a new home
    elif code == '404' or code.startswith('5'):
        dead.append((url, final, code))                # gone, not moved - needs a new source
    elif normalize(final) != normalize(url):
        moved.append((url, final, code))

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
