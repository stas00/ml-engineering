#!/usr/bin/env python
import pickle, glob, os, torch

SHOW_ALL   = False   # False: only collectives that hung; True: every collective
NUM_FRAMES = 2       # user frames to show: the actual call site + who called it

TORCH_DIR = os.path.dirname(torch.__file__)      # wherever torch actually lives
def is_user(frame):                              # skip torch's own collective plumbing
    return not frame["filename"].startswith(TORCH_DIR)

d = pickle.load(open(sorted(glob.glob("/tmp/fr*"))[0], "rb"))
for e in d["entries"]:
    if SHOW_ALL or e["state"] != "completed":
        print(e["state"], e["profiling_name"])
        user = [f for f in e["frames"] if is_user(f)]   # innermost first
        for f in user[:NUM_FRAMES]:                      # call site, then its caller
            print("   ", f["name"], os.path.basename(f["filename"]) + ":" + str(f["line"]))
