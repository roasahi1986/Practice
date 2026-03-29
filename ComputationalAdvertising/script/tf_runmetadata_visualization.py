# Copyright (C) 2023 lusyu1986@icloud.com

import sys
import re
import random

from absl import app
from absl import flags
from absl import logging

from matplotlib import colormaps
import pandas as pd
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string("runmetadata_file", None, "RunMetadata file path")
flags.DEFINE_integer("top_n", None, "Top N operations to show")

def get_runmetadata(runmetadata_file):
  run_metadata = tf.compat.v1.RunMetadata()
  with tf.io.gfile.GFile(runmetadata_file, 'rb') as f:
    run_metadata.ParseFromString(f.read())

  return run_metadata

def get_dataframe(run_metadata):
  # set start_ts to the minimum timestamp
  start_ts = sys.maxsize
  for device in run_metadata.step_stats.dev_stats:
    for node in device.node_stats:
      if node.all_start_nanos < start_ts:
        start_ts = node.all_start_nanos

  events = []
  for device in run_metadata.step_stats.dev_stats:
    device_name = device.device
    for node in device.node_stats:
      events.append(dict(
        start          = node.all_start_nanos - start_ts,
        before_op      = node.op_start_rel_nanos,
        in_op          = node.op_end_rel_nanos - node.op_start_rel_nanos,
        after_op       = node.all_end_rel_nanos - node.op_end_rel_nanos,
        duration       = node.all_end_rel_nanos,
        scheduled      = node.scheduled_nanos,
        name           = node.node_name,
        timeline_label = node.timeline_label,
      ))
  events = pd.DataFrame.from_dict(events)
  events.set_index('name', inplace = True)

  return events

def get_barh_figure(df):
  candidates = list(colormaps)
  # select a colormap randomly
  cmap = candidates[random.randint(0, len(candidates) - 1)]

  # set the colors for each column
  colors = dict(start = 'white', before_op = 'orange', in_op = 'green', after_op = 'red')
  return df.plot.barh(stacked = True, color = colors, colormap=cmap).get_figure()

def main(_):
  run_metadata = get_runmetadata(FLAGS.runmetadata_file)
  df = get_dataframe(run_metadata)

  df = df.sort_values(by = ['in_op'], ascending = False)
  if FLAGS.top_n:
    df = df.head(FLAGS.top_n)
  df = df.sort_values(by = ['start', 'duration'], ascending = False)

  meaningful_column = ['start', 'before_op', 'in_op', 'after_op']
  figure = get_barh_figure(df[meaningful_column])

  figure.savefig(re.sub(r'\.pb$', '.png', FLAGS.runmetadata_file))

if __name__ == '__main__':
  flags.mark_flags_as_required(['runmetadata_file'])
  app.run(main)
