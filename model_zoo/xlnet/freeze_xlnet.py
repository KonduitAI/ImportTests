import collections
import re

import tensorflow as tf
from tensorflow.python.tools import freeze_graph

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

# Use (from the xlnet repo) xlnet/run_classifier.py --do_train=True --do_eval=True --task_name=imdb --data_dir=C:\Temp\TF_Graphs\xlnet_cased_L-24_H-1024_A-16\aclImdb\ --output_dir=proc_data/imdb --model_dir=model --uncased=False --spiece_model_file=C:\Temp\TF_Graphs\xlnet_cased_L-24_H-1024_A-16\spiece.model --model_config_path=C:\Temp\TF_Graphs\xlnet_cased_L-24_H-1024_A-16\xlnet_config.json --init_checkpoint=C:\Temp\TF_Graphs\xlnet_cased_L-24_H-1024_A-16\xlnet_model.ckpt --max_seq_length=128 --eval_batch_size=8 --num_hosts=1 --num_core_per_host=1 --learning_rate=2e-5 -- train_steps=40 -- warmup_steps=10 --save_steps=50 --iterations=50
# Insert a call to freeze_xlnet right before return in xlnet.function_builder.get_classification_loss (from XLNet repo)
#

def freeze_xlnet(output_names, graph, base_dir = "C:\\Temp\\TF_Graphs\\xlnet_cased_L-24_H-1024_A-16", frozen_file='tf_model.pb'):
    print("Outputs: ", output_names)
    frozen_graph = freeze_session(tf.Session(), graph=graph, output_names=output_names)
    tf.train.write_graph(graph, base_dir, frozen_file + ".txt",
                         as_text=True)
    tf.train.write_graph(frozen_graph, base_dir, frozen_file, as_text=False)

if __name__ == '__main__':

    with tf.Session() as sess:

        base_dir = "C:\\Temp\\TF_Graphs\\xlnet_cased_L-24_H-1024_A-16"
        frozen_file = "tf_model.pb"

        checkpoint_path = "C:\\Temp\\TF_Graphs\\xlnet_cased_L-24_H-1024_A-16\\xlnet_model.ckpt"
        print(checkpoint_path)

        saver = tf.train.import_meta_graph(checkpoint_path + '.meta')

        for op in sess.graph.as_graph_def().node:
            print(op.name)

        names = sess.graph.get_tensor_by_name("save/SaveV2/tensor_names:0")

        print(sess.run(names))

        saver.restore(sess, checkpoint_path)
        print("Restored structure...")
        saver.restore(sess, checkpoint_path)
        print("Restored params...")

        # should be model/transformer/dropout_2
        frozen_graph = freeze_session(sess, output_names=["model/transformer/layer_23/ff/LayerNorm/batchnorm/add_1"])
        tf.train.write_graph(sess.graph, base_dir, frozen_file + ".txt",
                             as_text=True)
        tf.train.write_graph(frozen_graph, base_dir, frozen_file, as_text=False)


