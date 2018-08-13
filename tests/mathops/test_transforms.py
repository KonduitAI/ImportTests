import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph
from tfoptests.math_ops import DifferentiableMathOps


class MathTransform(TestGraph):
    def __init__(self, numInputs=1, *args, **kwargs):
        super(MathTransform, self).__init__(*args, **kwargs)
        self.numInputs = numInputs

    def list_inputs(self):
        if(self.numInputs == 1):
            return ["input_0"]
        else:
            return ["input_0", "input_1"]


def test_mathtransform():
    ops = [
        #Following values: already exist under transforms_0
        #"abs", "acos", "add", "ceil", "cos", "exp", "log", "max", "min"

        #Order here: [name, numInputs, broadcastable]
        #Order here: [name, numInputs, shape1, shape2]
        #["atan2", 2, [3,4], [3,4]],     #broadcastable
        #["atan2", 2, [3,1,4], [1,2,4]],
        #["div", 2, [3,4], [3,4]],       #Broadcastable
        #["div", 2, [3,4], [1,4]],
        #["div", 2, [3,1], [1,4]],
        #["div_scalar", 1, [3,4], None],    #Can't find this in TF docs... :/
        #["log_sigmoid", 1, [3,4], None],
        #All of these comparison ops support broadcasting...
        #["equal", 2, [3,4], [3,4]],
        #["equal", 2, [1,4], [3,4]],
        ["greater", 2, [3,4], [3,4]],
        ["greater", 2, [3,4], [4]],
        ["greater", 2, [3,1], [1,4]],
        ["greater", 2, [3,4], []],
        ["greater_equal", 2, [3,4], [3,4]],
        ["greater_equal", 2, [3,4], [4]],
        ["greater_equal", 2, [3,4], []],
        ["less", 2, [3,4], [3,4]],
        ["less", 2, [3,4], [4]],
        ["less", 2, [3,4], []],
        ["less_equal", 2, [3,4], [3,4]],
        ["less_equal", 2, [3,4], [4]],
        ["less_equal", 2, [3,4], []],
           ]




    for op in ops:
        tf.reset_default_graph()
        print("Running " + str(op))
        math_transform = MathTransform(seed=19,numInputs=op[1])
        if(len(op[2]) == 0):
            in_node_0 = tf.Variable(0.5, tf.float32)
        else:
            in_node_0 = tf.Variable(tf.random_normal(op[2]), tf.float32)
        if(op[1] > 1):
            if(len(op[3]) == 0):
                in_node_1 = tf.Variable(0.5, tf.float32)
            else:
                in_node_1 = tf.Variable(tf.random_normal(op[3]), tf.float32)
        else:
            in_node_1 = None
        constr = DifferentiableMathOps(in_node_0, in_node_1)
        answer = constr.execute(op[0])
        print(answer)
        constr.set_a(answer)

        placeholders = []

        if(answer.dtype == tf.bool):
            outNode = answer
        else:
            outNode = tf.add(answer, 1.0)

        predictions = [outNode]

        print()

        # Run and persist
        testName = "transforms/" + op[0] + "_" + ','.join(str(x) for x in op[2])
        if(op[1] > 1):
            testName = testName + "_" + ','.join(str(x) for x in op[3])
        tfp = TensorFlowPersistor(save_dir=testName)
        tfp.set_placeholders(placeholders) \
            .set_output_tensors(predictions) \
            .set_test_data(math_transform.get_test_data()) \
            .build_save_frozen_graph()

if __name__ == '__main__':
    test_mathtransform()
