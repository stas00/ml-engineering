#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# this script renames event names in tensorboard log files
# it does the rename in place (so make back ups!)
#
# example:
#
# find . -name "*.tfevents*" -exec tb-rename-events.py {} "iteration-time" "iteration-time/iteration-time" \;
#
# more than one old tag can be remapped to one new tag - use `;` as a separator:
#
# tb-rename-events.py events.out.tfevents.1 "training loss;validation loss" "loss"
#
# this script is derived from https://stackoverflow.com/a/60080531/9201239
#
# Important: this script requires CUDA environment.

import shlex
import sys
from pathlib import Path
import os
# avoid using the GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''
# disable logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.core.util.event_pb2 import Event

def rename_events(input_file, old_tags, new_tag):
    new_file = input_file + ".new"
    # Make a record writer
    with tf.io.TFRecordWriter(new_file) as writer:
        # Iterate event records
        for rec in tf.data.TFRecordDataset([input_file]):
            # Read event
            ev = Event()
            ev.MergeFromString(rec.numpy())
            # Check if it is a summary
            #print(ev)
            if ev.summary:
                # Iterate summary values
                for v in ev.summary.value:
                    #print(v)
                    # Check if the tag should be renamed
                    if v.tag in old_tags:
                        # Rename with new tag name
                        v.tag = new_tag
            writer.write(ev.SerializeToString())
    os.rename(new_file, input_file)

def rename_events_dir(input_file, old_tags, new_tag):
    # Write renamed events
    rename_events(input_file, old_tags, new_tag)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(f'{sys.argv[0]} <input file> <old tags> <new tag>',
              file=sys.stderr)
        sys.exit(1)
    input_file, old_tags, new_tag = sys.argv[1:]
    print(input_file, shlex.quote(old_tags), shlex.quote(new_tag))
    old_tags = old_tags.split(';')
    rename_events_dir(input_file, old_tags, new_tag)
