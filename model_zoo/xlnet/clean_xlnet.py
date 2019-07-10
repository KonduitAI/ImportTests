
import tensorflow as tf
from tensorflow import Graph
from tensorflow.tools.graph_transforms import TransformGraph

from model_zoo.util.freeze_keras import freeze_session

graphFile = "C:\\Temp\\TF_Graphs\\xlnet_cased_L-24_H-1024_A-16\\tf_model_old.pb"
base_dir = "C:\\Temp\\TF_Graphs\\xlnet_cased_L-24_H-1024_A-16"
frozen_file = "tf_model.pb"
if __name__ == '__main__':
    if graphFile is None:
        raise ValueError("Graph file location not set")

    with tf.gfile.GFile(graphFile, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    graph: Graph = tf.Graph()
    with tf.Graph().as_default():
        with tf.Session(graph=graph) as sess:
            tf.import_graph_def(graph_def, name="")

            # for op in graph.get_operations():
            #     print(op.name)

            transforms = ['strip_unused_nodes(type=int32, shape="8,128")']
            graph2 = TransformGraph(graph.as_graph_def(), inputs=[], outputs=["model_2/classification_imdb/logit/BiasAdd"], transforms=transforms)

            # graph2 = tf.graph_util.extract_sub_graph(graph.as_graph_def(), ["model_2/classification_imdb/logit/BiasAdd"])
            # tf.import_graph_def(graph2, name="")

            transpose = graph.get_operation_by_name("transpose")
            transpose_in = tf.placeholder('int32', (8, 128), "input")
            transpose._update_input(0, transpose_in)

            transpose1 = graph.get_operation_by_name("transpose_1")
            transpose1_in = tf.placeholder('int32', (8, 128), "input_1")
            transpose1._update_input(0, transpose1_in)

            transpose2 = graph.get_operation_by_name("transpose_2")
            transpose2_in = tf.placeholder('float32', (8, 128), "input_2")
            transpose2._update_input(0, transpose2_in)

            # transforms = ['strip_unused_nodes(type=int32, shape="8,128")']
            # graph2 = TransformGraph(graph.as_graph_def(), inputs=[], outputs=["model_2/classification_imdb/logit/BiasAdd"], transforms=transforms)

            frozen_graph = freeze_session(sess,
                                          output_names=["model_2/classification_imdb/logit/BiasAdd"])
            tf.train.write_graph(graph2, base_dir, frozen_file + ".txt",
                                 as_text=True)
            tf.train.write_graph(frozen_graph, base_dir, frozen_file, as_text=False)
