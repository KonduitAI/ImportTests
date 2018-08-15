import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph
from tfoptests.reduce_ops import ReduceOps
from tfoptests.ops import OpCreator
from tfoptests.var_initializer import VarInitializer


class OpTest(TestGraph):
    def __init__(self, op, *args, **kwargs):
        super(OpTest, self).__init__(*args, **kwargs)
        self.op = op

    def list_inputs(self):
        return self.op.get("phNames", [])

    def _get_placeholder_shape(self, name):
        '''Get input tensor shape for given node name'''
        return self.op.get("phShapes", {})

    def get_placeholder_input(self, name):
        '''Get input tensor for given node name'''
        return self.invals[name]

    def createVars(self, shapes, dtypes, init):
        print("Creating vars: shapes=", shapes, ", dtypes=", dtypes, ", init=", init)
        out = []
        initializer = VarInitializer()
        # for(s in shapes):
        for i in range(len(shapes)):
            s = shapes[i]
            d = tf.float32
            if(dtypes is not None):
                d = dtypes[i]

            n = "in_" + str(i)

            out.append(initializer.newVar(init[i], s, d, n))

        return out




def test_mathtransform():
    ops = [
        #Format:
        {"opName": "segment_max", "outName": "segment/segment_max_rank1", "varShapes":[[10], [10]], "varTypes":["float32", "int32"], "varInit":["range", "segment5"]}
           ]

    # max, mean, min, prod, sum



    for op in ops:
        tf.reset_default_graph()
        print("Running " + str(op))
        test = OpTest(seed=19, op=op)

        opName = op["opName"]
        varShapes = op.get("varShapes")
        varTypes = op.get("varTypes")
        varInit = op.get("varInit")
        phShapes = op.get("phShapes")

        opCreator = OpCreator(op)

        vars = test.createVars(varShapes, varTypes, varInit)
        opCreator.setVars(vars)

        out = opCreator.execute(opName)

        print(out)

        # Run and persist
        testName = op["outName"]
        tfp = TensorFlowPersistor(save_dir=testName)
        tfp.set_placeholders([]) \
            .set_output_tensors(out) \
            .set_test_data(test.get_test_data()) \
            .build_save_frozen_graph()


if __name__ == '__main__':
    test_mathtransform()
