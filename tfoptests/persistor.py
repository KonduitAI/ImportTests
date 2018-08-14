from __future__ import print_function

try:
    from itertools import izip as zip
except:
    # just use plain zip in py3
    pass
import sys
import os
import errno
import shutil
import tensorflow as tf
import numpy as np
from tensorflow.python.tools import freeze_graph

BASE_DIR = os.environ['DL4J_TEST_RESOURCES'] + '/src/main/resources/tf_graphs/examples'


class TensorFlowPersistor:
    '''
    TensorFlowPersistor (TFP) is the main abstraction of this module. A TFP
    has all the functionality to load and store tensorflow tests.

    TFP is an abstract base class. You need to implement `get_input`
    and `get_input_shape` for the graph data of your choice.
    '''

    def __init__(self, save_dir, base_dir=None, verbose=True):
        self.save_dir = save_dir
        self.base_dir = BASE_DIR if base_dir is None else base_dir
        self.verbose = verbose
        self._sess = None
        self._placeholders = None
        self._output_tensors = None
        self._placeholder_name_value_dict = None
        if not os.path.exists("{}/{}".format(self.base_dir, self.save_dir)):
            os.makedirs("{}/{}".format(self.base_dir, self.save_dir))

    def set_placeholders(self, graph_placeholders):
        self._placeholders = graph_placeholders
        return self

    def set_output_tensors(self, graph_output_tensors):
        '''
        TODO: Document after we decide on a framework structure
        '''
        self._output_tensors = graph_output_tensors
        return self

    def set_test_data(self, input_dict):
        self._placeholder_name_value_dict = input_dict
        return self

    def set_training_sess(self, sess):
        self._sess = sess
        return self

    def _write_to_file(self, nparray, content_file, shape_file):
        if np.isscalar(nparray):
            np.savetxt(shape_file, np.asarray([0]), fmt="%i")
            f = open(content_file, 'w')
            f.write('{}'.format(nparray))
            f.close()
        else:
            np.savetxt(shape_file, np.asarray(nparray.shape), fmt="%i")
            np.savetxt(content_file, np.ndarray.flatten(nparray), fmt="%10.8f")

    def _save_content(self, nparray, varname, name):
        content_file = "{}/{}/{}.{}.csv".format(self.base_dir, self.save_dir, varname, name)
        shape_file = "{}/{}/{}.{}.shape".format(self.base_dir, self.save_dir, varname, name)
        self._write_to_file(nparray, content_file, shape_file)

    def _save_input(self, nparray, varname, name='placeholder'):
        self._save_content(nparray, varname, name)

    def _save_intermediate(self, nparray, varname, name='prediction_inbw'):
        self._save_content(nparray, varname, name)

    def _save_prediction(self, output, varname='output', name='prediction'):
        self._save_content(output, varname, name)

    def _save_predictions(self, output_dict, name='prediction'):
        for output_name, output_value in output_dict.items():
            self._save_content(output_value, output_name, name)

    def _save_graph(self, sess, all_saver, data_path="data-all", model_file="model.txt"):
        all_saver.save(sess, "{}/{}/{}".format(self.base_dir, self.save_dir, data_path),
                       global_step=1000)
        tf.train.write_graph(sess.graph_def, "{}/{}".format(self.base_dir, self.save_dir),
                             model_file, True)

    def _freeze_n_save_graph(self, output_node_names="output",
                             restore_op_name="save/restore_all",
                             filename_tensor_name="save/Const:0"):
        try:
            checkpoint = tf.train.get_checkpoint_state("{}/{}/".format(self.base_dir, self.save_dir))
            input_checkpoint = checkpoint.model_checkpoint_path
        except:
            raise ValueError("Could not read checkpoint state for path {}/{}"
                             .format(self.base_dir, self.save_dir))
        if self.verbose:
            print(input_checkpoint)
        output_graph = "{}/{}/frozen_model.pb".format(self.base_dir, self.save_dir)
        input_graph = "{}/{}/model.txt".format(self.base_dir, self.save_dir)
        freeze_graph.freeze_graph(input_graph=input_graph,
                                  input_saver="",
                                  input_checkpoint=input_checkpoint,
                                  output_graph=output_graph,
                                  input_binary=False,
                                  output_node_names=output_node_names,
                                  restore_op_name=restore_op_name,
                                  filename_tensor_name=filename_tensor_name,
                                  clear_devices=True,
                                  initializer_nodes="")

    def write_frozen_graph_txt(self, model_file='frozen_model.pb'):
        graph_filename = "{}/{}/{}".format(self.base_dir, self.save_dir, model_file)
        with tf.gfile.GFile(graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.train.write_graph(graph_def, "{}/{}/".format(self.base_dir, self.save_dir),
                                 'frozen_graph.pbtxt', True)

    def load_frozen_graph(self, model_file='frozen_model.pb'):
        graph_filename = "{}/{}/{}".format(self.base_dir, self.save_dir, model_file)
        graph = tf.Graph()
        with tf.gfile.GFile(graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)
        return graph

    def _save_intermediate_nodes(self, input_dict):
        graph = self.load_frozen_graph()
        placeholder_dict = {}
        prediction_dict = {}
        if self.verbose:
            print("-----------------------------------------------------")
            print("PLACEHOLDER LIST:")
        for op in graph.get_operations():
            if op.type != "Placeholder":
                continue
            if self.verbose:
                print(op.name)  # there is a prefix and a suffix - there should only be one prefix
            placeholder_name = "/".join(op.name.split("/")[1:])
            placeholder_dict[op.name + ":0"] = input_dict[placeholder_name]
        if self.verbose:
            print("-----------------------------------------------------")

        for op in graph.get_operations():
            if op.type == "Placeholder":
                continue
            if self.verbose:
                print(op.name)
                print(op.type)
            for op_output in op.outputs:
                if self.verbose:
                    print(op_output.name)
                with tf.Session(graph=graph) as sess:
                    try:
                        op_prediction = sess.run(op_output, feed_dict=placeholder_dict)
                        tensor_output_name = ("/".join(op_output.name.split("/")[1:])).split(":")[0]
                        tensor_output_num = op_output.name.split(":")[1]
                        if tensor_output_name in self._list_output_node_names():
                            prediction_dict[tensor_output_name] = op_prediction
                        if self.verbose:
                            print(op_prediction)
                            print("-----------------------------------------------------")
                        modified_tensor_output_name = "____".join(tensor_output_name.split("/"))
                        save_to = ".".join([modified_tensor_output_name, tensor_output_num])
                        self._save_intermediate(op_prediction, save_to)
                    except:
                        print("Unexpected error:", sys.exc_info()[0])
                        if self.verbose:
                            print(op_output)
                            print("SKIPPING")
                            print("-----------------------------------------------------")
        return prediction_dict

    def load_external_graph(self, model_file):
        graph = tf.Graph()
        graph_def = tf.GraphDef()

        with open(model_file, "rb") as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)

        return graph

    def _get_placeholder_dict(self):
        placeholder_feed_dict = {}
        for input_tensor in self._placeholders:
            input_name = input_tensor.name.split(":")[0]
            input_value = self._placeholder_name_value_dict[input_name]
            placeholder_feed_dict[input_tensor] = input_value
            self._save_input(input_value, input_name)
        return placeholder_feed_dict

    def _check_outputs(self):
        if self._output_tensors is None:
            raise ValueError("Ouput tensor list not set")
        for a_output in self._output_tensors:
            if isinstance(a_output, list):
                raise ValueError('Output tensor elements cannot be lists...')

    def _check_inputs(self):
        if self._placeholders is None:
            raise ValueError("Input tensor placeholder list not set")

    def _clean_dir(self):
        working_dir = "{}/{}".format(self.base_dir, self.save_dir)
        if os.path.exists(working_dir):
            shutil.rmtree(working_dir)
        try:
            os.makedirs(working_dir)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

    def _list_output_node_names(self):
        output_node_names = []
        for a_output in self._output_tensors:
            output_node_names.append(a_output.name.split(":")[0])
        return output_node_names

    def build_save_frozen_graph(self, skip_intermediate=False):
        self._check_inputs()  # make sure input placeholders are set
        self._check_outputs()  # make sure outputs are set
        self._clean_dir()  # clean contents of dir
        placeholder_feed_dict = self._get_placeholder_dict()
        all_saver = tf.train.Saver()
        if self._sess is None:
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                predictions = sess.run(self._output_tensors, feed_dict=placeholder_feed_dict)
                self._save_graph(sess, all_saver)
        else:
            predictions = self._sess.run(self._output_tensors, feed_dict=placeholder_feed_dict)
            self._save_graph(self._sess, all_saver)
            self._sess.close
        first_pass_dict = dict(zip(self._list_output_node_names(), predictions))
        if self.verbose:
            print(predictions)
        self._save_predictions(first_pass_dict)
        tf.reset_default_graph()
        self._freeze_n_save_graph(output_node_names=",".join(self._list_output_node_names()))
        self.write_frozen_graph_txt()
        if not skip_intermediate:
            second_pass_dict = self._save_intermediate_nodes(self._placeholder_name_value_dict)
            # TODO: Better way to do this assert??
            assert second_pass_dict.keys() == first_pass_dict.keys()
            for a_output in second_pass_dict.keys():
                np.testing.assert_equal(first_pass_dict[a_output], second_pass_dict[a_output])
        return predictions
