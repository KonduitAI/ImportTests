import tensorflow as tf
import tensorflow_hub as hub
from tensorflow._api.v1 import graph_util

# Loading from hub doesn't work, but most of what you would need is here.

def freeze_graph(model_folder, output_nodes,
                 output_filename='frozen.pb',
                 rename_outputs=None):
    # Load checkpoint
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path

    output_graph = output_filename

    # Devices should be cleared to allow Tensorflow to control placement of
    # graph when loading on different machines
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta',
                                       clear_devices=True)

    graph = tf.get_default_graph()

    onames = output_nodes

    # https://stackoverflow.com/a/34399966/4190475
    if rename_outputs is not None:
        nnames = rename_outputs.split(',')
        with graph.as_default():
            for o, n in zip(onames, nnames):
                _out = tf.identity(graph.get_tensor_by_name(o + ':0'), name=n)
            onames = nnames

    input_graph_def = graph.as_graph_def()

    # fix batch norm nodes
    for node in input_graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']

    with tf.Session(graph=graph) as sess:
        saver.restore(sess, input_checkpoint)

        # In production, graph weights no longer need to be updated
        # graph_util provides utility to change all variables to constants
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, input_graph_def,
            onames  # unrelated nodes will be discarded
        )

        # Serialize and write to file
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

def load_from_hub(name, module, base_path="C:\\Temp\\TF_Graphs", signature=None):
    tf.reset_default_graph()
    module = hub.Module(module)
    inputs = {k: tf.placeholder(v.dtype, v.get_shape().as_list(), k)
              for k, v in module.get_input_info_dict().items()}

    if signature is not None:
        output = module(inputs, signature=signature)
    else:
        output = module(inputs)

    initializer = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(initializer)

    graph = tf.get_default_graph()
    saver = tf.train.Saver()

    checkpoint = base_path + "\\" + name + "\\" + name

    saver.save(sess, checkpoint)
    tf.train.write_graph(graph, base_path + "\\" + name, "model.pb", False)
    tf.train.write_graph(graph, base_path + "\\" + name, "model.pb.txt", True)

    print()
    print("Inputs:", [k + ": " + v.name for k, v in inputs.items()])
    print("Output:", output.name)
    print("Initializer:", initializer.name)

    freeze_graph(base_path + "\\" + name,
                 output_filename=base_path + "\\" + name + "\\" + "frozen.pb",
                 output_nodes=[output.name.split(":")[0]])

    return module

if __name__ == '__main__':
    m = load_from_hub("compare_gan_model_1_celebahq128_resnet19",
                      "https://tfhub.dev/google/compare_gan/model_1_celebahq128_resnet19/1",
                      signature="generator")