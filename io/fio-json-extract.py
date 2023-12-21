#!/bin/env python

#
# usage:
#
# ./fio-json-extract.py fio-json-file.json
#
# The script expects an fio generated json file as the only input that is `filename.json` from
# `fio ... --output-format=json --output=filename.json`
#
# The will print out a markdown table of average latency, bandwidth and iops

import io, json, sys

if len(sys.argv) != 2:
    raise ValueError("usage: ./fio-json-extract.py fio-json-file.json")

with open(sys.argv[1], 'r') as f:
    d = json.load(f)

# expects a single job output
job = d['jobs'][0]
rw_type = job['jobname'] # read | write
section = job[rw_type]
numjobs = int(d['global options']['numjobs'])

headers = ["lat msec", "bw MBps", "  IOPS  ", "jobs"]
width = [len(h) for h in headers]

print("| " + " | ".join(headers)  + " |")

print(f"| {'-'*(width[0]-1)}: | {'-'*(width[1]-1)}: | {'-'*(width[2]-1)}: | {'-'*(width[3]-1)}: | ")

print(f"| {section['lat_ns']['mean']/10**6:{width[1]}.1f} | {section['bw_bytes']/2**20:{width[0]}.1f} | {int(section['iops']):{width[2]}d} | {numjobs:{width[3]}d} |")
