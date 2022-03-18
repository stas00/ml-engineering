#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# this script removes events from tensorboard log files by specific tag names
# it does the removal in place (so make back ups!)
#
# example:
#
#  find . -name "*.tfevents*" -exec tb-remove-events-by-tag.py {} "batch-size/batch-size" \;
#
# more than one tag can be removed - use `;` as a separator:
#
#  tb-remove-events-by-tag.py events.out.tfevents.1 "batch-size/batch-size;batch-size/batch-size vs samples"
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

def remove_events(input_file, tags_to_remove):
    new_file = input_file + ".new"
    # Make a record writer
    with tf.io.TFRecordWriter(new_file) as writer:
        # Iterate event records
        for rec in tf.data.TFRecordDataset([input_file]):
            # Read event
            ev = Event()
            ev.MergeFromString(rec.numpy())
            # Check if it is a summary event
            if ev.summary:
                orig_values = [v for v in ev.summary.value]
                filtered_values = [v for v in orig_values if v.tag not in tags_to_remove]
                #print(f"filtered_values={len(filtered_values)}, orig_values={len(orig_values)}")
                if len(filtered_values) != len(orig_values):
                    # for v in orig_values:
                    #     print(v)
                    del ev.summary.value[:]
                    ev.summary.value.extend(filtered_values)
            writer.write(ev.SerializeToString())
    os.rename(new_file, input_file)

def remove_events_dir(input_file, tags_to_remove):
    # Write removed events
    remove_events(input_file, tags_to_remove)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f'{sys.argv[0]} <input file> <tags to remove>',
              file=sys.stderr)
        sys.exit(1)
    input_file, tags_to_remove = sys.argv[1:]
    print(input_file, shlex.quote(tags_to_remove))
    tags_to_remove = tags_to_remove.split(';')
    remove_events_dir(input_file, tags_to_remove)
