#!/usr/bin/env python3
"""Report external links whose URL has moved or died, so the book can cite the endpoint.

A redirect is invisible to a reader and to build/check-links.py - the old URL still works,
so nothing looks broken - yet it means the book names a location the project has left.
GitHub org renames are the common case: outlines-dev/outlines, TimDettmers/bitsandbytes.

Two mechanisms are checked. HTTP redirects, which curl follows for us, and <meta refresh>,
which it does not - that one is markup, so the stale URL answers 200 and looks healthy to every
other check here. Detecting it costs one extra request per HTML page, which roughly doubles the
runtime; --no-meta skips it when only the fast HTTP pass is wanted.

Needs network, so this is not part of the fast local pass - run it deliberately.

Politeness: requests to one domain are serialized with --delay seconds between them, while
different domains proceed in parallel. Never remove this. The book cites 243 distinct
github.com URLs and 46 on huggingface.co; firing those off concurrently looks like a scraper
and gets the runner throttled or IP-blocked, which is far more expensive than a slow check.
(largest per-domain count x delay) is only a per-domain floor, not the run time. Measured
2026-08-07: 697 URLs across 128 domains took 3h24m at the defaults, against a 12-minute
floor - so budget hours, not minutes. Meta-refresh probing adds up to MAX_META_HOPS more
delayed requests per URL, which contributes, but the full breakdown was not measured.
Raise --jobs to cover more domains at once, not to go faster on one.

usage: python build/check-redirects.py [--jobs N] [--delay SECS] [--no-meta] [file ...]
       (defaults to chapters-md.txt)
"""
import re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin, urlparse

URL = re.compile(r'https?://[^)\]}"\'>\s]+')
TRAILING = '.,;:!?`'

# A URL inside a `code span` is a template for the reader to complete, not a citation. Concrete
# failure: debug/tools.md says to visit `github.com/huggingface/transformers/commit/` "and append
# the commit SHA", and this was reported as having moved to /commit/main - a "fix" that would have
# broken the instruction. Only 2 of the book's URLs live solely in code spans and both are that
# example, so stripping them costs no coverage.
#
# Note this deliberately does NOT skip ``` fences ```, unlike build/check-links.py: 13 URLs appear
# only inside fences and they include live download links such as the Miniconda installer and
# dcgm-exporter, whose death would break the book's setup instructions. Those must stay checked.
CODE_SPAN = re.compile(r'`[^`]*`')

# Some vendors reject a bare HTTP client outright - amd.com, hpe.com, microsoft.com, nasa.gov
# all do. Retrying with a browser user-agent separates real 404s from bot mitigation, which
# matters: amd.com's instinct/specifications.html looked merely "blocked" for a whole pass
# and was in fact gone. See SESSION.md "Sources and citations" item 9.
# Keep this a current, complete browser string. Bot mitigation fingerprints the whole token
# sequence, so an old or truncated UA - this was a 2015 Firefox 39 string, and a Chrome one with
# "(KHTML, like Gecko)" left out also failed - reads as automation and gets gated anyway.
BROWSER_UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
              '(KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36')

# Hosts that redirect to signed, expiring, or geo-specific endpoints. The final URL is not a
# canonical location and must never be pasted into the book - it rots within hours and pins
# the reader to one region's CDN.
CDN_HOSTS = ('cdn.hf.co', 'cloudfront.net', 'akamaized.net', 'blob.core.windows.net')

# A <meta http-equiv="refresh"> is a redirect that curl does not follow, because it is markup
# rather than an HTTP status. The old URL keeps answering 200 forever, so a relocated page looks
# perfectly healthy to the checks above. Concrete failure: the book cited
# docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html, which answers 200 with a
# 1KB stub that refreshes to user_guide/torch_compiler/, which refreshes again to the real page -
# two hops, both invisible. Chains are why MAX_META_HOPS is not 1.
META_TAG = re.compile(r"""<meta[^>]+http-equiv\s*=\s*["']?refresh["']?[^>]*>""", re.I)
META_URL = re.compile(r"""url\s*=\s*["']?([^"'>\s;]+)""", re.I)
META_DELAY = re.compile(r"""content\s*=\s*["']?\s*(\d+)""", re.I)
MAX_META_HOPS = 3

# A 403/429/503 carrying a large HTML body is usually a JavaScript browser check rather than
# throttling or an outage. Concrete failure: hud.pytorch.org/benchmark/compilers answers 429 with
# 33KB of "Vercel Security Checkpoint - Enable JavaScript to continue". That was read as
# rate-limiting and retried with delays and a browser user-agent, none of which can work - the
# gate wants a JS engine, not patience. Worth its own bucket because the right action is the
# opposite of the unreachable bucket's: do not retry, do not slow down, and do not call it dead,
# since a reader with a browser reaches the page normally.
CHALLENGE = re.compile(r'Vercel Security Checkpoint|Just a moment\.\.\.|cf-browser-verification|'
                       r'challenge-platform|Enable JavaScript to continue|Attention Required!', re.I)

# Some meta-refresh targets are a JS gate's retry endpoint carrying a single-use session token,
# not a new home. Concrete failure: google.com/search?q=... refreshes to
# /httpservice/retry/enablejs?sei=<token>, which the first version of this check offered as the
# replacement URL - it would have pasted a dead session token into the book.
JS_RETRY = re.compile(r'/httpservice/retry/enablejs|/cdn-cgi/challenge|__cf_chl|/sorry/index', re.I)

# Illustrative URLs in the prose, not citations - there is nothing to check or fix.
LOCAL_HOST = re.compile(r'^https?://(localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\])([:/]|$)', re.I)

# Docs sites commonly refresh a rolling alias to the release it currently points at. The target is
# correct today and wrong next release, so it must never be pasted into the book - cite the alias.
# Same hazard as CDN_HOSTS, different mechanism.
ROLLING_ALIAS = re.compile(r'/(?:stable|latest|current|main|master)/', re.I)

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

def cms_landing(url, final):
    """True when a bare domain root redirects to a path on the same host. That path is a CMS
    landing route - beegfs.io/ serves its home page at /c/, wandb.ai/ at /site - and the root is
    both the friendlier citation and the more durable one, since the route will be renamed long
    before the domain is. A root redirecting to a *different* host is a real move and is still
    reported."""
    p = urlparse(url)
    return p.path in ('', '/') and not p.query and urlparse(final).hostname == p.hostname

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
           '-w', '%{url_effective}\t%{http_code}\t%{content_type}']
    if ua:
        cmd += ['-A', ua]
    try:
        r = subprocess.run(cmd + [url], capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return None, 'timeout', ''
    out = r.stdout.strip().split('\t')
    return (out[0], out[1], out[2]) if len(out) == 3 else (None, 'error', '')

def resolve(url, delay):
    """A bare client first, then a browser user-agent if the host refused to answer at all."""
    final, code, ctype = final_url(url)
    if final is None or code in ('000', 'error', 'timeout'):
        time.sleep(delay)
        final, code, ctype = final_url(url, ua=BROWSER_UA)
    return final, code, ctype

def head_bytes(url, n=8192):
    """The first n bytes of the body. A meta refresh has to be in <head> to work at all, so this
    is always enough, and the range request stops a stray tarball being pulled down in full."""
    try:
        r = subprocess.run(['curl', '-sL', '--max-time', '20', '-A', BROWSER_UA,
                            '-H', f'Range: bytes=0-{n - 1}', url],
                           capture_output=True, timeout=30)
    except subprocess.TimeoutExpired:
        return ''
    return r.stdout[:n].decode('utf-8', 'replace')

def meta_target(url, body):
    """Absolute URL that a meta refresh in `body` points at, or None if there isn't one."""
    tag = META_TAG.search(body)
    if not tag:
        return None
    tag = tag.group(0)
    wait = META_DELAY.search(tag)
    # A long delay is a human-facing "this page has moved" notice, not a relocation to chase.
    if wait and int(wait.group(1)) > 10:
        return None
    target = META_URL.search(tag)
    if not target:
        return None
    target = urljoin(url, target.group(1).strip())
    return target if normalize(target) != normalize(url) else None

def dealias(src, target):
    """When a rolling alias refreshes to a pinned release, neither URL is the one to cite. The
    target pins the book to a release; the source may also predate a reorganization, as
    /stable/torch.compiler_troubleshooting.html did before the page moved under user_guide/. So
    keep the alias and take the rest of the move: /stable/x.html + /2.13/user_guide/y.html gives
    /stable/user_guide/y.html, which follows the reorganization without pinning the version."""
    alias = ROLLING_ALIAS.search(src)
    if not alias:
        return None
    out = re.sub(r'/(?:v?\d+\.\d+(?:\.\d+)?)/', alias.group(0), target, count=1)
    return out if out != target else None

def follow_meta(url, code, ctype, delay, enabled=True):
    """Chase a meta-refresh chain from an otherwise-healthy page. Returns where it ends up and
    the hops taken, so a two-hop stub chain is reported as one move for the reader."""
    if not enabled or code != '200' or 'html' not in (ctype or '').lower():
        return url, []
    chain, cur = [], url
    for _ in range(MAX_META_HOPS):
        time.sleep(delay)
        nxt = meta_target(cur, head_bytes(cur))
        if not nxt:
            break
        chain.append(nxt)
        cur = nxt
    return cur, chain

def suggested(url, final):
    """curl never sends the #fragment, so `final` always lacks it. Carry it over, or a naive
    replace silently downgrades a deep link to its page - and can collide with a sibling
    fragment on the same page, mangling the URL outright."""
    if '#' not in url:
        return final, ''
    frag = url.split('#', 1)[1]
    return (f'{final.split("#")[0]}#{frag}',
            '  (fragment carried over - verify it still exists on the new page)')

DEFAULT_JOBS, DEFAULT_DELAY = 8, 3.0
USAGE = ('usage: python build/check-redirects.py [--jobs N] [--delay SECS] [--no-meta] [file ...]\n'
         f'  --jobs N     domains to check concurrently (default {DEFAULT_JOBS}); raises domain\n'
         '               coverage, never the rate on one domain\n'
         f'  --delay SECS seconds between requests to the same domain (default {DEFAULT_DELAY});\n'
         '               may be raised, never lowered - see SESSION.md External link rot item 2\n'
         '  --no-meta    skip meta-refresh probing; HTTP redirects only, much faster\n'
         '  file ...     chapters to check (default: every file in chapters-md.txt)\n'
         '\nA full run takes hours and prints nothing until it finishes. Unknown options are an\n'
         'error rather than being ignored, so that a typo cannot start a multi-hour sweep.')

def die(msg):
    print(f'{msg}\n\n{USAGE}', file=sys.stderr)
    sys.exit(2)

args, jobs, delay, do_meta = [], DEFAULT_JOBS, DEFAULT_DELAY, True
argv = sys.argv[1:]
i = 0
while i < len(argv):
    a = argv[i]
    if a in ('--help', '-h'):
        print(USAGE)
        sys.exit(0)
    elif a == '--no-meta':
        do_meta = False
    elif a.partition('=')[0] in ('--jobs', '--delay'):
        # accept both --opt=value and --opt value; the space form is what the usage documents,
        # and silently ignoring it used to leave the value behind as a bogus filename.
        # Match on the exact name before '=' - a prefix test let --jobss=4 through and ran.
        name, eq, raw = a.partition('=')
        if not eq:
            i += 1
            if i >= len(argv):
                die(f'{name} needs a value')
            raw = argv[i]
        elif not raw:
            die(f'{name} needs a value')
        try:
            val = int(raw) if name == '--jobs' else float(raw)
        except ValueError:
            die(f'{name} needs a {"whole number" if name == "--jobs" else "number"}, got {raw!r}')
        if val <= 0:
            die(f'{name} must be positive, got {val}')
        if name == '--jobs':
            jobs = val
        else:
            if val < DEFAULT_DELAY:
                die(f'--delay {val} is below the {DEFAULT_DELAY}s default. The delay may be raised '
                    f'but never lowered -\nlowering it looks like a scraper and gets the runner '
                    f'blocked. See SESSION.md External link rot item 2.')
            delay = val
    elif a.startswith('-'):
        die(f'unknown option: {a}')
    else:
        args.append(a)
    i += 1

files = args or [l.strip() for l in open('chapters-md.txt') if l.strip()]

sites = {}                                             # url -> [(file, line), ...]
for f in files:
    try:
        lines = open(f, encoding='utf-8').read().split('\n')
    except OSError:
        print(f'MISSING CHAPTER {f}')
        continue
    for ln, line in enumerate(lines, 1):
        for u in extract(CODE_SPAN.sub('', line)):
            if LOCAL_HOST.match(u):            # an example in the prose, not a citation
                continue
            sites.setdefault(u, []).append((f, ln))

by_host = {}
for u in sites:
    by_host.setdefault(urlparse(u).hostname or '', []).append(u)

worst = max((len(v) for v in by_host.values()), default=0)
print(f'checking {len(sites)} distinct external URLs across {len(by_host)} domains from '
      f'{len(files)} files\n{jobs} domains at a time, {delay}s between requests to the same '
      f'domain - the busiest has {worst} URLs, so the per-domain floor alone is '
      f'~{int(worst * delay * (2 if do_meta else 1) / 60)} min'
      f'{"" if do_meta else " (--no-meta: HTTP redirects only)"}\n'
      f'that floor is not the run time: 697 URLs / 128 domains took 3h24m at these defaults '
      f'on 2026-08-07, so budget hours\n'
      f'nothing is printed until the run finishes - an empty log is not a hang\n')

def check_host(host):
    """One domain's URLs, sequentially, spaced by `delay`. Called once per domain so that
    different domains overlap while a single domain is never hit concurrently."""
    out = []
    for i, u in enumerate(by_host[host]):
        if i:
            time.sleep(delay)
        final, code, ctype = resolve(u, delay)
        # chase from where HTTP left off, not from the original URL
        meta_final, chain = follow_meta(final or u, code, ctype, delay, do_meta)
        out.append((u, final, code, meta_final, chain))
    return out

results = []
with ThreadPoolExecutor(max_workers=jobs) as pool:
    for chunk in pool.map(check_host, by_host):
        results.extend(chunk)

moved, unreachable, cdn, dead, meta, challenged = [], [], [], [], [], []
for url, final, code, meta_final, chain in results:
    if final is None or code in ('000', 'error', 'timeout'):
        unreachable.append((url, code))
    elif any(h in final for h in CDN_HOSTS):
        cdn.append(url)                                # a download endpoint, not a new home
    elif code in ('403', '429') and CHALLENGE.search(head_bytes(final)):
        challenged.append((url, code))                 # a JS gate - see CHALLENGE above
    elif code == '404' or code.startswith('5'):
        dead.append((url, final, code))                # gone, not moved - needs a new source
    else:
        if normalize(final) != normalize(url) and not cms_landing(url, final):
            moved.append((url, final, code))
        if chain:                                      # invisible to HTTP - see META_TAG above
            if JS_RETRY.search(meta_final):
                challenged.append((url, 'js-gate'))    # a retry token, not a home
            else:
                fix = suggested(url, dealias(url, meta_final) or meta_final)[0]
                # A rolling alias that refreshes only to its pinned release needs no change: the
                # book already cites the right URL. Reporting those buried the 3 real findings
                # under 30 no-ops on the first full run.
                if normalize(fix) != normalize(url):
                    meta.append((url, meta_final, chain, fix))

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

if meta:
    print(f'\n{len(meta)} URL(s) answer 200 but the page has MOVED via <meta refresh> - the old '
          f'URL will keep working indefinitely, so this is invisible to every other check here. '
          f'Rolling aliases that merely refresh to their current release are not listed, having '
          f'nothing to change:')
    for url, target, chain, fix in sorted(meta):
        where = sites[url][0]
        print(f'  {where[0]}:{where[1]}')
        print(f'    from {url}')
        for i, hop in enumerate(chain, 1):
            print(f'    hop {i} {hop}')
        if dealias(url, target):
            print(f'    NOT the raw target - that pins the book to one release; the alias is kept '
                  f'and the rest of the move applied. Verify it resolves:')
        print(f'    suggest {fix}')

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

if challenged:
    print(f'\n{len(challenged)} URL(s) sit behind a JavaScript browser check. The page is fine in '
          f'a browser and the link is not broken - do NOT retry, slow down, or swap user-agent, '
          f'and do NOT quote content from these without opening them yourself:')
    for url, code in sorted(challenged):
        where = sites[url][0]
        print(f'  [{code}] {where[0]}:{where[1]}  {url}')

if unreachable:
    print(f'\n{len(unreachable)} URL(s) could not be checked '
          f'(blocked, rate-limited, or offline) - not necessarily dead:')
    for url, code in sorted(unreachable):
        print(f'  [{code}] {url}')

print(f'\n{len(sites)} checked: {len(moved)} moved, {len(meta)} meta-refresh, {len(dead)} dead, '
      f'{len(cdn)} CDN, {len(challenged)} JS-gated, {len(unreachable)} unreachable')
sys.exit(1 if moved or dead or meta else 0)
