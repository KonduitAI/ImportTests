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
        #Format: [opName, testName, inputShapes, inputTypes, axis, extra, random_init, placeholder]
        # ["reduce_sum", "sum_0", [[3,4]], None, [0], {"keepdims":False}],
        # ["reduce_sum", "sum_1keep", [[3,4]], None, [1], {"keepdims":True}],
        # ["reduce_sum", "sum_01", [[3,4]], None, [0,1], {"keepdims":False}],
        # ["reduce_sum", "sum_012keep", [[3,4,5]], None, [0,1,2], {"keepdims":True}],
        # ["reduce_sum", "sum_all", [[3,4,5]], None, None, {"keepdims":False}],
        # ["reduce_sum", "sum_scalar", [[]], None, None, {"keepdims":False}],
        # ["reduce_max", "max_0", [[3,4]], None, [0], {"keepdims":False}],
        # ["reduce_max", "max_1keep", [[3,4]], None, [1], {"keepdims":True}],
        # ["reduce_max", "max_01", [[3,4]], None, [0,1], {"keepdims":False}],
        # ["reduce_max", "max_012keep", [[3,4,5]], None, [0,1,2], {"keepdims":True}],
        # ["reduce_max", "max_all", [[3,4,5]], None, None, {"keepdims":False}],
        # ["reduce_max", "max_scalar", [[]], None, None, {"keepdims":False}],
        # ["reduce_min", "min_0", [[3,4]], None, [0], {"keepdims":False}],
        # ["reduce_min", "min_1keep", [[3,4]], None, [1], {"keepdims":True}],
        # ["reduce_min", "min_01", [[3,4]], None, [0,1], {"keepdims":False}],
        # ["reduce_min", "min_012keep", [[3,4,5]], None, [0,1,2], {"keepdims":True}],
        # ["reduce_min", "min_all", [[3,4,5]], None, None, {"keepdims":False}],
        # ["reduce_min", "min_scalar", [[]], None, None, {"keepdims":False}],
        # ["reduce_mean", "mean_1keep", [[3,4]], None, [1], {"keepdims":True}],
        # ["reduce_mean", "mean_01", [[3,4]], None, [0,1], {"keepdims":False}],
        # ["reduce_mean", "mean_012keep", [[3,4,5]], None, [0,1,2], {"keepdims":True}],
        # ["reduce_mean", "mean_all", [[3,4,5]], None, None, {"keepdims":False}],
        # ["reduce_mean", "mean_scalar", [[]], None, None, {"keepdims":False}],
        # ["reduce_prod", "prod_1keep", [[3,4]], None, [1], {"keepdims":True}],
        # ["reduce_prod", "prod_01", [[3,4]], None, [0,1], {"keepdims":False}],
        # ["reduce_prod", "prod_012keep", [[3,4,5]], None, [0,1,2], {"keepdims":True}],
        # ["reduce_prod", "prod_all", [[3,4,5]], None, None, {"keepdims":False}],
        # ["reduce_prod", "prod_scalar", [[]], None, None, {"keepdims":False}]

        # ["argmax", "argmax3,4_0", [[3,4]], None, 0, None],
        # ["argmax", "argmax3,4_1", [[3,4]], None, 1, None],
        # ["argmax", "argmax3,4_-2", [[3,4]], None, -2, None],
        # ["argmax", "argmax3,4,5_-1", [[3,4,5]], None, -1, None],
        # ["argmin", "argmin3,4_0", [[3,4]], None, 0, None],
        # ["argmin", "argmin3,4_1", [[3,4]], None, 1, None],
        # ["argmin", "argmin3,4_-1", [[3,4]], None, -1, None],
        # ["argmin", "argmin3,4,5_-2", [[3,4,5]], None, -2, None],

        # ["add_n", "add_n", [[3,4], [3,4], [3,4]], None, None, None],
        # ["add_n", "add_n_single", [[3,4]], None, None, None],
        # ["add_n", "add_n_single_scalar", [[]], None, None, None]

        #Problem here: ref should be a variable, indices/updates should be placeholder not variable
        #Order of args: ref, indices, updates
        #["scatter_add", "scatter_add_scalar", [[10,3], [], [3]], [tf.float32, tf.int32, tf.float32], None, None]


        #Can't execute these: FileNotFoundError: [Errno 2] No such file or directory: 'C:\\DL4J\\Git\\dl4j-test-resources/src/main/resources/tf_graphs/examples/reductions/normalize_moments/normalize/mean.prediction.shape'
        #["normalize_moments", "normalize_moments", [[], [5], [5]], [tf.float32, tf.float32, tf.float32], None, ["uniform10", None, "uniform"]],  #Args: count, mean_ss, variance_ss, shift
        #["normalize_moments", "normalize_moments_shift", [[], [5], [5], []], [tf.float32, tf.float32, tf.float32, tf.float32], None, None, ["uniform10", None, "uniform", "uniform"]]

        #Can't execute these:
        # ["count_nonzero", "count_nonzero_0", [[3,4]], None, [0], {"keepdims":False}],
        # ["count_nonzero", "count_nonzero_1", [[3,4]], None, [1], {"keepdims":False}],
        # ["count_nonzero", "count_nonzero_1keep", [[3,4]], None, [1], {"keepdims":True}],
        # ["count_nonzero", "count_nonzero_all", [[3,4,5]], None, None, {"keepdims":False}],
        # ["count_nonzero", "count_nonzero_scalar", [[]], None, None, {"keepdims":False}],
        # ["count_nonzero", "count_nonzero_345_-1", [[3,4,5]], None, [-1], {"keepdims":False}],

        #Also having issues with these: FileNotFoundError: [Errno 2] No such file or directory: 'C:\\DL4J\\Git\\dl4j-test-resources/src/main/resources/tf_graphs/examples/reductions/moments0/moments/Squeeze.prediction.shape'
        # ["moments", "moments0", [[3,4]], None, [0], {"keepdims":False} ],
        # ["moments", "moments1", [[3,4]], None, [1], {"keepdims":False} ],
        # ["moments", "moments01", [[3,4]], None, [0,1], {"keepdims":False} ],
        # ["moments", "moments1keep", [[3,4]], None, [1], {"keepdims":True} ],
        # ["moments", "moments345-02", [[3,4,5]], None, [0,2], {"keepdims":False} ],
        # ["moments", "moments2345-023", [[2,3,4,5]], None, [0,2,3], {"keepdims":True} ]

        #Can't execute these
        #["is_non_decreasing", "is_non_decreasing_3-4", [[3,4]], None, None, None],
        #["is_non_decreasing", "is_non_decreasing_scalar", [[]], None, None, None]
           ]

    # max, mean, min, prod, sum



    for op in ops:
        tf.reset_default_graph()
        print("Running " + str(op))
        math_transform = Reductions(seed=19, numInputs=len(op[2]))

        # print("op[2]: ", op[2])
        # print("op[3]: ", op[3])

        init = None
        if(len(op) > 6):
            init = op[6]

        vars = getVars(op[2], op[3], init)

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

def getVars(shapes, dtypes, init):
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
            set = False
            if(init is not None):
                if(len(init) > i and init[i] is not None):
                    if(init[i] == "uniform"):
                        out.append(tf.Variable(tf.random_uniform(s), tf.float32))
                        set = True
                    elif(init[i] == "uniform10"):
                        out.append(tf.Variable(tf.random_uniform(s, minval=0.0, maxval=10.0), tf.float32))
                        set = True
            if(set != True):
                out.append(tf.Variable(tf.random_normal(s), tf.float32))
        elif(d == tf.int32):
            out.append(tf.Variable(tf.random_uniform(s, minval=1, maxval=10, dtype=tf.int32)))
        else:
            raise Exception("Datatype not implemented")

    return out


if __name__ == '__main__':
    test_mathtransform()
