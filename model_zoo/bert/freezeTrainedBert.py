import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.tools.graph_transforms import TransformGraph

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

            '''
            # input_names = ["IteratorGetNext"]
            input_names = ["IteratorGetNext:0", "IteratorGetNext:1", "IteratorGetNext:4"]
            output_names = ["loss/LogSoftmax"]
            transforms = ['strip_unused_nodes(type=int32, shape="4,128")']
            graph2 = TransformGraph(graph2.as_graph_def(), input_names, output_names, transforms)
            # graph2 = TransformGraph(graph2, input_names, output_names, transforms)

            # graph2 = tf.graph_util.remove_training_nodes(input_graph=graph2.as_graph_def())
            graph2 = tf.graph_util.remove_training_nodes(input_graph=graph2)
            '''

            '''
            input_names = ["IteratorGetNext:0", "IteratorGetNext:1", "IteratorGetNext:4"]
            output_names = ["loss/LogSoftmax"]
            transforms = ['strip_unused_nodes(type=int32, shape="4,128")']
            # graph2 = TransformGraph(graph2.as_graph_def(), input_names, output_names, transforms)
            graph2 = TransformGraph(graph2.as_graph_def(), inputs=input_names, outputs=output_names, transforms=transforms)
            '''

            '''
            #2019-02-27 00:36:31.079753: I tensorflow/tools/graph_transforms/transform_graph.cc:317] Applying strip_unused_nodes
            # terminate called after throwing an instance of 'std::out_of_range'
            # what():  map::at
            # Aborted
            input_names = ["IteratorV2"]    #Same result with "IteratorV2:0"
            output_names = ["loss/LogSoftmax"]
            transforms = ['strip_unused_nodes(type=resource)']
            graph2 = TransformGraph(graph2.as_graph_def(), inputs=input_names, outputs=output_names, transforms=transforms)
            '''

            # input_names = ["IteratorGetNext", "IteratorGetNext:1", "IteratorGetNext:4"]
            input_names = []
            # output_names = ["loss/LogSoftmax"]
            output_names = ["loss/Softmax"]
            transforms = ['strip_unused_nodes(type=int32, shape="4,128")']
            # graph2 = TransformGraph(graph2.as_graph_def(), input_names, output_names, transforms)
            graph2 = TransformGraph(graph2.as_graph_def(), inputs=input_names, outputs=output_names, transforms=transforms)

            # for op in graph2.get_operations():
            #     print(op.name)
            return graph2


dirIn = "/TF_Graphs/mrpc_output/"
dirOut = dirIn + "frozen/"
ckpt = "model.ckpt-2751"
graph = load_graph(dirIn + ckpt)
txtPath = dirOut + "bert_export.pb.txt"
tf.train.write_graph(graph, dirOut, "bert_export.pb.txt", True)

output_graph = dirOut + "bert_mrpc_frozen.pb"
print("Freezing Graph...")
freeze_graph.freeze_graph(
    input_graph=txtPath,
    input_checkpoint=dirIn+ckpt,
    input_saver="",
    output_graph=output_graph,
    input_binary=False,
    # output_node_names="loss/LogSoftmax",     #This is log(prob(x))
    output_node_names="loss/Softmax",     #This is log(prob(x))
    restore_op_name="save/restore_all",
    filename_tensor_name="save/Const:0",
    clear_devices=True,
    initializer_nodes="")
print("Freezing graph complete...")