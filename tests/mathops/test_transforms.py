import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph
from tfoptests.math_ops import DifferentiableMathOps


class MathTransform(TestGraph):
    def __init__(self, numInputs=1, *args, **kwargs):
        super(MathTransform, self).__init__(*args, **kwargs)
        self.input_0 = np.random.uniform(size=(3, 3))
        if(numInputs == 2):
            self.input_1 = np.random.uniform(size=(3, 3)) + np.random.uniform(size=(3, 3))
        self.numInputs = numInputs

    def list_inputs(self):
        if(self.numInputs == 1):
            return ["input_0"]
        else:
            return ["input_0", "input_1"]

    def get_placeholder_input(self, name):
        if name == "input_0":
            return self.input_0
        if name == "input_1":
            return self.input_1

    def _get_placeholder_shape(self, name):
        if name == "input_0" or name == "input_1":
            return [3, 3]


def test_mathtransform():
    ops = [
        #Following values: already exist under transforms_0
        #"abs", "acos", "add", "ceil", "cos", "exp", "log", "max", "min"
        ["log_sigmoid", 1]

           # , "add_n"
           # , "cross"
           # , "log1p"
           # , "mod"
           # , "mathmul"
           # , "cumprod"
           # , "cumsum"
           # , "erf"
           # , "count_nonzero"
           # , "greater"
           # , "greater_equal"
           # , "equal"
           ]




    for op in ops:
        print("Running " + str(op))
        math_transform = MathTransform(seed=19,numInputs=op[1])
        in_node_0 = math_transform.get_placeholder("input_0", data_type=tf.float32)
        if(op[1] > 1):
            in_node_1 = math_transform.get_placeholder("input_1", data_type=tf.float32)
        else:
            in_node_1 = None
        k0 = tf.Variable(tf.random_normal([8, 8]), name="in0")
        constr = DifferentiableMathOps(in_node_0, in_node_1)
        answer = constr.execute(op[0])
        print(answer)
        constr.set_a(answer)

        if(op[1] > 1):
            placeholders = [in_node_0, in_node_1]
        else:
            placeholders = [in_node_0]
        predictions = [answer]

        # Run and persist
        tfp = TensorFlowPersistor(save_dir="transforms/" + op[0])
        tfp.set_placeholders(placeholders) \
            .set_output_tensors(predictions) \
            .set_test_data(math_transform.get_test_data()) \
            .build_save_frozen_graph()

if __name__ == '__main__':
    test_mathtransform()
