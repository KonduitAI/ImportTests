import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph
from tfoptests.math_ops import DifferentiableMathOps


class Reductions(TestGraph):
    def __init__(self, numInputs=1, *args, **kwargs):
        super(Reductions, self).__init__(*args, **kwargs)
        self.numInputs = numInputs
        self.innames = ["input_" + str(i) for i in range(numInputs)]

    def list_inputs(self):
        return innames


def test_mathtransform():
    ops = [
        #Format: [opName, testName, inputShapes, inputTypes, axis, extra]
        ["sum", "sum_0", [[3,4]], None, [0], {"keepdims":False}]
           ]




    for op in ops:
        tf.reset_default_graph()
        print("Running " + str(op))
        math_transform = Reductions(seed=19, numInputs=len(op[2]))

        vars = getVars(op[2], op[3])

        placeholders = []

        reduction = ReduceOps(vars, op[2], op[3], op[4], op[5])

        print()

        # Run and persist
        testName = "reductions/" + op[1]
        tfp = TensorFlowPersistor(save_dir=testName)
        tfp.set_placeholders(placeholders) \
            .set_output_tensors(reduction) \
            .set_test_data(math_transform.get_test_data()) \
            .build_save_frozen_graph()

def getVars(self, shapes, dtypes=None):
    print("shapes", shapes)
    print("dtypes", dtypes)
    out = []
    # for(s in shapes):
    for i in range(len(shapes)):
        s = shapes[i]
        d = tf.float32
        if(dtypes is not None):
            d = dtypes[i]

        if(d == tf.bool):
            out.append(tf.Variable(tf.random_normal(s) >= 0, tf.bool))
        elif(d == tf.float32):
            out.append(tf.Variable(tf.random_normal(s), tf.float32))
        else:
            raise Exception("Datatype not implemented")

    return out


if __name__ == '__main__':
    test_mathtransform()
