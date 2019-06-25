from sys import argv

from keras import backend as K, Model
import tensorflow as tf
from keras.engine.saving import load_model


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

def freeze_keras_model(model: Model, base_dir, frozen_file):
    frozen_graph = freeze_session(K.get_session(),
                                  output_names=[out.op.name for out in
                                                model.outputs])
    tf.train.write_graph(frozen_graph, base_dir, frozen_file, as_text=False)
    tf.train.write_graph(frozen_graph, base_dir, frozen_file + ".txt", as_text=True)

def freeze_keras_file(keras_file, tf_dir, tf_file = "tf_model.pb"):
    model = load_model(keras_file)

    print("Inputs:", model.inputs)
    print("Outputs:", model.outputs)

    freeze_keras_model(model, tf_dir, tf_file)

if __name__ == '__main__':
    # freeze_keras_file("C:\\Users\\jimne\\Google Drive\\Poly Stuff\\CSC 490\\Lab 4\\PorV.h5",
    #                   "C:\\Temp\\TF_Graphs\\" + "PorVRNN")
    freeze_keras_file(
        "C:\\Users\\jimne\\Desktop\\NN Server\\generator_model.h5",
        "C:\\Temp\\TF_Graphs\\" + "cifar10_gan_85")
    # freeze_keras_file(
    #     "C:\\Users\\jimne\\Downloads\\text-gen.h5",
    #     "C:\\Temp\\TF_Graphs\\" + "text_gen_81")
