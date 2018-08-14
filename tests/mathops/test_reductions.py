import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph
from tfoptests.reduce_ops import ReduceOps


class Reductions(TestGraph):
    def __init__(self, numInputs=1, *args, **kwargs):
        super(Reductions, self).__init__(*args, **kwargs)
        self.numInputs = numInputs
        self.innames = ["input_" + str(i) for i in range(numInputs)]

    def list_inputs(self):
        return self.innames


def test_mathtransform():
    ops = [
        #Format: [opName, testName, inputShapes, inputTypes, axis, extra]
        ["reduce_sum", "sum_0", [[3,4]], None, [0], {"keepdims":False}],
        ["reduce_sum", "sum_1keep", [[3,4]], None, [1], {"keepdims":True}],
        ["reduce_sum", "sum_01", [[3,4]], None, [0,1], {"keepdims":False}],
        ["reduce_sum", "sum_012keep", [[3,4,5]], None, [0,1,2], {"keepdims":True}],
        ["reduce_sum", "sum_all", [[3,4,5]], None, None, {"keepdims":False}],
        ["reduce_sum", "sum_scalar", [[]], None, None, {"keepdims":False}],
        ["reduce_max", "max_0", [[3,4]], None, [0], {"keepdims":False}],
        ["reduce_max", "max_1keep", [[3,4]], None, [1], {"keepdims":True}],
        ["reduce_max", "max_01", [[3,4]], None, [0,1], {"keepdims":False}],
        ["reduce_max", "max_012keep", [[3,4,5]], None, [0,1,2], {"keepdims":True}],
        ["reduce_max", "max_all", [[3,4,5]], None, None, {"keepdims":False}],
        ["reduce_max", "max_scalar", [[]], None, None, {"keepdims":False}],
        ["reduce_min", "min_0", [[3,4]], None, [0], {"keepdims":False}],
        ["reduce_min", "min_1keep", [[3,4]], None, [1], {"keepdims":True}],
        ["reduce_min", "min_01", [[3,4]], None, [0,1], {"keepdims":False}],
        ["reduce_min", "min_012keep", [[3,4,5]], None, [0,1,2], {"keepdims":True}],
        ["reduce_min", "min_all", [[3,4,5]], None, None, {"keepdims":False}],
        ["reduce_min", "min_scalar", [[]], None, None, {"keepdims":False}],
        ["reduce_mean", "mean_1keep", [[3,4]], None, [1], {"keepdims":True}],
        ["reduce_mean", "mean_01", [[3,4]], None, [0,1], {"keepdims":False}],
        ["reduce_mean", "mean_012keep", [[3,4,5]], None, [0,1,2], {"keepdims":True}],
        ["reduce_mean", "mean_all", [[3,4,5]], None, None, {"keepdims":False}],
        ["reduce_mean", "mean_scalar", [[]], None, None, {"keepdims":False}],
        ["reduce_prod", "prod_1keep", [[3,4]], None, [1], {"keepdims":True}],
        ["reduce_prod", "prod_01", [[3,4]], None, [0,1], {"keepdims":False}],
        ["reduce_prod", "prod_012keep", [[3,4,5]], None, [0,1,2], {"keepdims":True}],
        ["reduce_prod", "prod_all", [[3,4,5]], None, None, {"keepdims":False}],
        ["reduce_prod", "prod_scalar", [[]], None, None, {"keepdims":False}]
           ]

    # max, mean, min, prod, sum



    for op in ops:
        tf.reset_default_graph()
        print("Running " + str(op))
        math_transform = Reductions(seed=19, numInputs=len(op[2]))

        print("op[2]: ", op[2])
        print("op[3]: ", op[3])
        vars = getVars(op[2], op[3])

        placeholders = []

        reduction = ReduceOps(vars, op[2], op[3], op[4], op[5])
        out = reduction.execute(op[0])

        print(out)

        # Run and persist
        testName = "reductions/" + op[1]
        tfp = TensorFlowPersistor(save_dir=testName)
        tfp.set_placeholders(placeholders) \
            .set_output_tensors(out) \
            .set_test_data(math_transform.get_test_data()) \
            .build_save_frozen_graph()

def getVars(shapes, dtypes):
    print("shapes: ", shapes)
    print("dtypes: ", dtypes)
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
