import tensorflow as tf
from tensorflow.python.tools import freeze_graph

def load_graph(checkpoint_path):
    init_all_op = tf.initialize_all_variables()
    graph2 = tf.Graph()
    with graph2.as_default():
        with tf.Session(graph=graph2) as sess:
            saver = tf.train.import_meta_graph(checkpoint_path + '.meta')
            saver.restore(sess, checkpoint_path)
            print("Restored structure...")
            saver.restore(sess, checkpoint_path)
            print("Restored params...")

            # for op in graph2.get_operations():
            #     print(op.name)
            return graph2


dir = "/TF_Graphs/trainEvalBertOutput/"
graph = load_graph(dir + "model.ckpt-2")
txtPath = dir + "bert.pb.txt"
tf.train.write_graph(graph, dir, "bert.pb.txt", True)

output_graph = dir + "FROZEN.pb"
print("Freezing Graph...")
freeze_graph.freeze_graph(
    input_graph=txtPath,
    input_checkpoint=dir+"model.ckpt-2",
    input_saver="",
    output_graph=output_graph,
    input_binary=False,
    output_node_names="loss/LogSoftmax",     #This is log(prob(x))
    restore_op_name="save/restore_all",
    filename_tensor_name="save/Const:0",
    clear_devices=True,
    initializer_nodes="")
print("Freezing graph complete...")