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

            varInit = "uniform"
            if(init is not None and init[i] is not None):
                varInit = init[i]

            out.append(initializer.newVar(varInit, s, d, n))

        return out

    def createPlaceholders(self, shapes, dtypes, init):
        print("Creating vars: shapes=", shapes, ", dtypes=", dtypes, ", init=", init)
        out = []
        initializer = VarInitializer()
        for i in range(len(shapes)):
            s = shapes[i]
            d = tf.float32
            if(dtypes is not None):
                d = dtypes[i]

            n = "in_ph_" + str(i)

            varInit = "uniform"
            if(init is not None and init[i] is not None):
                varInit = init[i]

            out.append(initializer.newPlaceholder(varInit, s, d, n))

        return out




def test_mathtransform():
    ops = [
        #Format:
        # {"opName": "segment_max", "outName": "segment/segment_max_rank1", "varShapes":[[20], [20]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "segment5"]},
        # {"opName": "segment_mean", "outName": "segment/segment_mean_rank1", "varShapes":[[20], [20]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "segment5"]},
        # {"opName": "segment_min", "outName": "segment/segment_min_rank1", "varShapes":[[20], [20]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "segment5"]},
        # {"opName": "segment_prod", "outName": "segment/segment_prod_rank1", "varShapes":[[20], [20]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "segment5"]},
        # {"opName": "segment_sum", "outName": "segment/segment_sum_rank1", "varShapes":[[20], [20]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "segment5"]},
        # {"opName": "segment_max", "outName": "segment/segment_max_rank2", "varShapes":[[6,3], [6]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "segment3"]},
        # {"opName": "segment_mean", "outName": "segment/segment_mean_rank2", "varShapes":[[6,3], [6]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "segment3"]},
        # {"opName": "segment_min", "outName": "segment/segment_min_rank2", "varShapes":[[6,3], [6]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "segment3"]},
        # {"opName": "segment_prod", "outName": "segment/segment_prod_rank2", "varShapes":[[6,3], [6]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "segment3"]},
        # {"opName": "segment_sum", "outName": "segment/segment_sum_rank2", "varShapes":[[6,3], [6]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "segment3"]},
        # {"opName": "segment_max", "outName": "segment/segment_max_rank3", "varShapes":[[6,3,2], [6]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "segment3"]},
        # {"opName": "segment_mean", "outName": "segment/segment_mean_rank3", "varShapes":[[6,3,2], [6]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "segment3"]},
        # {"opName": "segment_min", "outName": "segment/segment_min_rank3", "varShapes":[[6,3,2], [6]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "segment3"]},
        # {"opName": "segment_prod", "outName": "segment/segment_prod_rank3", "varShapes":[[6,3,2], [6]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "segment3"]},
        # {"opName": "segment_sum", "outName": "segment/segment_sum_rank3", "varShapes":[[6,3,2], [6]], "varTypes":["float32", "int32"], "varInit":["uniform_int10", "segment3"]},
        # {"opName": "space_to_batch", "outName": "space_to_batch/rank4nhwc", "varShapes":[[2,4,4,4], [2,2]], "varTypes":["float32", "int32"], "varInit":["range", "zero"]},
        # {"opName": "space_to_batch", "outName": "space_to_batch/rank4nhwc_pad", "varShapes":[[2,2,2,4], [2,2]], "varTypes":["float32", "int32"], "varInit":["range", "one"]},
        # {"opName": "space_to_depth", "outName": "space_to_depth/rank4nhwc", "varShapes":[[2,4,4,4]], "varTypes":["float32", "int32"], "varInit":["range", "zero"], "data_format":"NHWC"},
        # {"opName": "space_to_depth", "outName": "space_to_depth/rank4nchw", "varShapes":[[2,4,4,4]], "varTypes":["float32", "int32"], "varInit":["range", "zero"], "data_format":"NCHW"},
        # {"opName": "batch_to_space", "outName": "batch_to_space/rank4nhwc", "varShapes":[[8,2,2,4], [2,2]], "varTypes":["float32", "int32"], "varInit":["range", "zero"]},
        # {"opName": "batch_to_space", "outName": "batch_to_space/rank4nhwc_crop", "varShapes":[[8,2,2,4], [2,2]], "varTypes":["float32", "int32"], "varInit":["range", "one"]},
        # {"opName": "depth_to_space", "outName": "depth_to_space/rank4nhwc", "varShapes":[[2,4,4,4]], "varTypes":["float32", "int32"], "varInit":["range", "zero"], "data_format":"NHWC"},
        #{"opName": "depth_to_space", "outName": "depth_to_space/rank4nchw", "varShapes":[[2,4,4,4]], "varTypes":["float32", "int32"], "varInit":["range", "zero"], "data_format":"NCHW"},  #Only NHWC format supported on CPU!?
        {"opName": "size", "outName": "size_rank2", "varShapes":[[3,4]], "varTypes":["float32"]},
        {"opName": "size", "outName": "size_rank3", "varShapes":[[2,3,4]], "varTypes":["float32"]},
        {"opName": "shape", "outName": "shape_rank2", "varShapes":[[3,4]], "varTypes":["float32"]},
        {"opName": "shape", "outName": "shape_rank3", "varShapes":[[2,3,4]], "varTypes":["float32"]}
        # {"opName": "shapen", "outName": "shapen_3x2", "varShapes":[[3,4], [1,2], [2,4]], "varTypes":["float32", "float32", "float32"]},
        # {"opName": "shapen", "outName": "shapen_3x3", "varShapes":[[2,3,4], [1,2,3], [2,1,2]], "varTypes":["float32", "float32", "float32"]}
        # {"opName": "matrix_inverse", "outName": "matrix_inverse/rank2", "varShapes":[[3,3]], "varTypes":["float32"], "varInit":["uniform"]},
        # {"opName": "matrix_inverse", "outName": "matrix_inverse/rank3", "varShapes":[[2,3,3]], "varTypes":["float32"], "varInit":["uniform"]},
        # {"opName": "matrix_inverse", "outName": "matrix_inverse/rank4", "varShapes":[[2,2,3,3]], "varTypes":["float32"], "varInit":["uniform"]}
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
        phTypes = op.get("phTypes")
        phInit = op.get("phInit")

        opCreator = OpCreator(op)

        vars = test.createVars(varShapes, varTypes, varInit)
        #ph = test.createPlaceholders(phShapes, phTypes, phInit)
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
