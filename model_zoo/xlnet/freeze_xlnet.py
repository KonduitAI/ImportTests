import collections
import re

import tensorflow as tf

from model_zoo.util.freeze_keras import freeze_session


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    # tf.logging.info('original name: %s', name)
    if name not in name_to_variable:
      continue
    # assignment_map[name] = name
    assignment_map[name] = name_to_variable[name]
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

  return (assignment_map, initialized_variable_names)

with tf.Session() as sess:

    base_dir = "C:\\Temp\\TF_Graphs\\xlnet_cased_L-24_H-1024_A-16"
    frozen_file = "tf_model.pb"

    checkpoint_path = "C:\\Temp\\TF_Graphs\\xlnet_cased_L-24_H-1024_A-16\\xlnet_model.ckpt"
    print(checkpoint_path)

    saver = tf.train.import_meta_graph(checkpoint_path + '.meta')
    saver.restore(sess, checkpoint_path)
    print("Restored structure...")
    saver.restore(sess, checkpoint_path)
    print("Restored params...")

    frozen_graph = freeze_session(sess)
    tf.train.write_graph(sess.graph, base_dir, frozen_file + ".txt",
                         as_text=True)
    tf.train.write_graph(frozen_graph, base_dir, frozen_file, as_text=False)