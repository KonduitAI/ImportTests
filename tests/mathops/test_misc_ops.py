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
        # {"opName": "size", "outName": "size_rank2", "varShapes":[[3,4]], "varTypes":["float32"]},
        # {"opName": "size", "outName": "size_rank3", "varShapes":[[2,3,4]], "varTypes":["float32"]},
        # {"opName": "shape", "outName": "shape_rank2", "varShapes":[[3,4]], "varTypes":["float32"]},
        # {"opName": "shape", "outName": "shape_rank3", "varShapes":[[2,3,4]], "varTypes":["float32"]}
        # {"opName": "shapen", "outName": "shapen_3x2", "varShapes":[[3,4], [1,2], [2,4]], "varTypes":["float32", "float32", "float32"]},
        # {"opName": "shapen", "outName": "shapen_3x3", "varShapes":[[2,3,4], [1,2,3], [2,1,2]], "varTypes":["float32", "float32", "float32"]}
        # {"opName": "matrix_inverse", "outName": "matrix_inverse/rank2", "varShapes":[[3,3]], "varTypes":["float32"], "varInit":["uniform"]},
        # {"opName": "matrix_inverse", "outName": "matrix_inverse/rank3", "varShapes":[[2,3,3]], "varTypes":["float32"], "varInit":["uniform"]},
        # {"opName": "matrix_inverse", "outName": "matrix_inverse/rank4", "varShapes":[[2,2,3,3]], "varTypes":["float32"], "varInit":["uniform"]}
        # {"opName": "pad", "outName": "pad/rank1Pzero_const0", "varShapes":[[5],[1,2],[]], "varTypes":["float32", "int32", "float32"], "varInit":["uniform","zero","zero"], "mode":"CONSTANT"},
        # {"opName": "pad", "outName": "pad/rank1Pzero_const10", "varShapes":[[5],[1,2],[]], "varTypes":["float32", "int32", "float32"], "varInit":["uniform","zero","ten"], "mode":"CONSTANT"},
        # {"opName": "pad", "outName": "pad/rank1Pone_const0", "varShapes":[[5],[1,2],[]], "varTypes":["float32", "int32", "float32"], "varInit":["uniform","one","zero"], "mode":"CONSTANT"},
        # {"opName": "pad", "outName": "pad/rank1Pone_const10", "varShapes":[[5],[1,2],[]], "varTypes":["float32", "int32", "float32"], "varInit":["uniform","one","ten"], "mode":"CONSTANT"},
        # {"opName": "pad", "outName": "pad/rank1Pone_reflect", "varShapes":[[5],[1,2]], "varTypes":["float32", "int32"], "varInit":["uniform","one"], "mode":"REFLECT"},
        # {"opName": "pad", "outName": "pad/rank1Pone_symmetric", "varShapes":[[5],[1,2]], "varTypes":["float32", "int32"], "varInit":["uniform","one"], "mode":"SYMMETRIC"}
        # {"opName": "pad", "outName": "pad/rank2Pzero_const0", "varShapes":[[3,4],[2,2],[]], "varTypes":["float32", "int32", "float32"], "varInit":["uniform","zero","zero"], "mode":"CONSTANT"},
        # {"opName": "pad", "outName": "pad/rank2Pzero_const10", "varShapes":[[3,4],[2,2],[]], "varTypes":["float32", "int32", "float32"], "varInit":["uniform","zero","ten"], "mode":"CONSTANT"},
        # {"opName": "pad", "outName": "pad/rank2Pone_const0", "varShapes":[[3,4],[2,2],[]], "varTypes":["float32", "int32", "float32"], "varInit":["uniform","one","zero"], "mode":"CONSTANT"},
        # {"opName": "pad", "outName": "pad/rank2Pone_const10", "varShapes":[[3,4],[2,2],[]], "varTypes":["float32", "int32", "float32"], "varInit":["uniform","one","ten"], "mode":"CONSTANT"},
        # {"opName": "pad", "outName": "pad/rank2Pone_reflect", "varShapes":[[3,4],[2,2]], "varTypes":["float32", "int32"], "varInit":["uniform","one"], "mode":"REFLECT"},
        # {"opName": "pad", "outName": "pad/rank2Pone_symmetric", "varShapes":[[3,4],[2,2]], "varTypes":["float32", "int32"], "varInit":["uniform","one"], "mode":"SYMMETRIC"},
        # {"opName": "pad", "outName": "pad/rank3Pzero_const0", "varShapes":[[2,3,4],[3,2],[]], "varTypes":["float32", "int32", "float32"], "varInit":["uniform","zero","zero"], "mode":"CONSTANT"},
        # {"opName": "pad", "outName": "pad/rank3Pzero_const10", "varShapes":[[2,3,4],[3,2],[]], "varTypes":["float32", "int32", "float32"], "varInit":["uniform","zero","ten"], "mode":"CONSTANT"},
        # {"opName": "pad", "outName": "pad/rank3Pone_const0", "varShapes":[[2,3,4],[3,2],[]], "varTypes":["float32", "int32", "float32"], "varInit":["uniform","one","zero"], "mode":"CONSTANT"},
        # {"opName": "pad", "outName": "pad/rank3Pone_const10", "varShapes":[[2,3,4],[3,2],[]], "varTypes":["float32", "int32", "float32"], "varInit":["uniform","one","ten"], "mode":"CONSTANT"},
        # {"opName": "pad", "outName": "pad/rank3Pone_reflect", "varShapes":[[2,3,4],[3,2]], "varTypes":["float32", "int32"], "varInit":["uniform","one"], "mode":"REFLECT"},
        # {"opName": "pad", "outName": "pad/rank3Pone_symmetric", "varShapes":[[2,3,4],[3,2]], "varTypes":["float32", "int32"], "varInit":["uniform","one"], "mode":"SYMMETRIC"},
        # {"opName": "unique", "outName": "unique10-5", "varShapes":[[10]], "varTypes":["float32"], "varInit":["uniform_int5"]},
        # {"opName": "unique_with_counts", "outName": "uniqueWithCounts10-5", "varShapes":[[10]], "varTypes":["float32"], "varInit":["uniform_int5"]},
        # {"opName": "topk", "outName": "topk/rank1_k1", "varShapes":[[10]], "varTypes":["float32"], "varInit":["uniform"], "k":1, "sorted":False},
        # {"opName": "topk", "outName": "topk/rank1_k1_sorted", "varShapes":[[10]], "varTypes":["float32"], "varInit":["uniform"], "k":1, "sorted": True},
        # {"opName": "topk", "outName": "topk/rank1_k5", "varShapes":[[10]], "varTypes":["float32"], "varInit":["uniform"], "k":5, "sorted":False},
        # {"opName": "topk", "outName": "topk/rank1_k5_sorted", "varShapes":[[10]], "varTypes":["float32"], "varInit":["uniform"], "k":5, "sorted":True},
        # {"opName": "topk", "outName": "topk/rank2_k1", "varShapes":[[3,6]], "varTypes":["float32"], "varInit":["uniform"], "k":1, "sorted":False},
        # {"opName": "topk", "outName": "topk/rank2_k1_sorted", "varShapes":[[3,6]], "varTypes":["float32"], "varInit":["uniform"], "k":1, "sorted": True},
        # {"opName": "topk", "outName": "topk/rank2_k5", "varShapes":[[3,6]], "varTypes":["float32"], "varInit":["uniform"], "k":5, "sorted":False},
        # {"opName": "topk", "outName": "topk/rank2_k5_sorted", "varShapes":[[3,6]], "varTypes":["float32"], "varInit":["uniform"], "k":5, "sorted":True},
        # {"opName": "topk", "outName": "topk/rank3_k3", "varShapes":[[3,4,5]], "varTypes":["float32"], "varInit":["uniform"], "k":3, "sorted":False},
        # {"opName": "topk", "outName": "topk/rank3_k3_sorted", "varShapes":[[3,4,5]], "varTypes":["float32"], "varInit":["uniform"], "k":3, "sorted":True}
        # {"opName": "in_top_k", "outName": "in_top_k/test_4,5_k1", "varShapes":[[4,5], [4]], "varTypes":["float32", "int32"], "varInit":["uniform", "uniform_int5"], "k":1},
        # {"opName": "in_top_k", "outName": "in_top_k/test_4,5_k3", "varShapes":[[4,5], [4]], "varTypes":["float32", "int32"], "varInit":["uniform", "uniform_int5"], "k":3}
        # {"opName": "matrix_determinant", "outName": "matrix_determinant/rank2_5,5", "varShapes":[[5,5]], "varTypes":["float32"], "varInit":["uniform"]},
        # {"opName": "matrix_determinant", "outName": "matrix_determinant/rank3_2,3,3", "varShapes":[[2,3,3]], "varTypes":["float32"], "varInit":["uniform"]},
        # {"opName": "matrix_determinant", "outName": "matrix_determinant/rank4_2,2,3,3", "varShapes":[[2,2,3,3]], "varTypes":["float32"], "varInit":["uniform"]},
        # {"opName": "matrix_set_diag", "outName": "matrix_set_diag/rank2_5,5", "varShapes":[[5,5], [5]], "varTypes":["float32", "float32"], "varInit":["zeros", "uniform"]},
        # {"opName": "matrix_set_diag", "outName": "matrix_set_diag/rank2_5,4", "varShapes":[[5,4], [4]], "varTypes":["float32", "float32"], "varInit":["zeros", "uniform"]},
        # {"opName": "matrix_set_diag", "outName": "matrix_set_diag/rank2_4,5", "varShapes":[[5,4], [4]], "varTypes":["float32", "float32"], "varInit":["zeros", "uniform"]},
        # {"opName": "matrix_set_diag", "outName": "matrix_set_diag/rank3_2,3,3", "varShapes":[[2,3,3], [2,3]], "varTypes":["float32", "float32"], "varInit":["zeros", "uniform"]},
        # {"opName": "matrix_set_diag", "outName": "matrix_set_diag/rank3_2,3,4", "varShapes":[[2,3,4], [2,3]], "varTypes":["float32", "float32"], "varInit":["zeros", "uniform"]},
        # {"opName": "matrix_set_diag", "outName": "matrix_set_diag/rank4_2,2,3,3", "varShapes":[[2,2,3,3], [2,2,3]], "varTypes":["float32", "float32"], "varInit":["zeros", "uniform"]}
        # {"opName": "identity_n", "outName": "identity_n_2", "varShapes":[[2,3], [2]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"]},
        # {"opName": "identity_n", "outName": "identity_n_4", "varShapes":[[2,3], [2], [], [2,1,3]], "varTypes":["float32", "float32", "float32", "float32"], "varInit":["uniform", "uniform", "uniform", "uniform"]}
        # {"opName": "zeta", "outName": "zeta_rank2", "varShapes":[[2,3], [2,3]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"]},
        # {"opName": "zeta", "outName": "zeta_rank3", "varShapes":[[2,3,2], [2,3,2]], "varTypes":["float32", "float32", "float32"], "varInit":["uniform", "uniform", "uniformt"]},
        # {"opName": "confusion_matrix", "outName": "confusion/no_num_classes", "varShapes":[[5], [5]], "varTypes":["int32", "int32"], "varInit":["uniform_int5", "uniform_int5"], "num_classes":None},
        # {"opName": "confusion_matrix", "outName": "confusion/with_num_classes", "varShapes":[[5], [5]], "varTypes":["int32", "int32"], "varInit":["uniform_int5", "uniform_int5"], "num_classes":5},
        # {"opName": "confusion_matrix", "outName": "confusion/no_num_classes_with_weights", "varShapes":[[5], [5], [5]], "varTypes":["int32", "int32", "float32"], "varInit":["uniform_int5", "uniform_int5", "uniform"], "num_classes":None},
        # {"opName": "confusion_matrix", "outName": "confusion/with_num_classes_with_weights", "varShapes":[[5], [5], [5]], "varTypes":["int32", "int32", "float32"], "varInit":["uniform_int5", "uniform_int5", "uniform"], "num_classes":5}
        # {"opName": "stack", "outName": "stack/rank0_axis-1", "varShapes":[[], []], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "axis":-1},
        # {"opName": "stack", "outName": "stack/rank0_axis0", "varShapes":[[], []], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "axis":0},
        # {"opName": "stack", "outName": "stack/rank1_axis-2", "varShapes":[[3], [3]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "axis":-2},
        # {"opName": "stack", "outName": "stack/rank1_axis-1", "varShapes":[[3], [3]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "axis":-1},
        # {"opName": "stack", "outName": "stack/rank1_axis-0", "varShapes":[[3], [3]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "axis":0},
        # {"opName": "stack", "outName": "stack/rank1_axis1", "varShapes":[[3], [3]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "axis":1},
        # {"opName": "stack", "outName": "stack/rank2_axis-3", "varShapes":[[2,3], [2,3]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "axis":-3},
        # {"opName": "stack", "outName": "stack/rank2_axis-2", "varShapes":[[2,3], [2,3]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "axis":-2},
        # {"opName": "stack", "outName": "stack/rank2_axis-1", "varShapes":[[2,3], [2,3]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "axis":-1},
        # {"opName": "stack", "outName": "stack/rank2_axis-0", "varShapes":[[2,3], [2,3]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "axis":0},
        # {"opName": "stack", "outName": "stack/rank2_axis1", "varShapes":[[2,3], [2,3]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "axis":1},
        # {"opName": "stack", "outName": "stack/rank2_axis2", "varShapes":[[2,3], [2,3]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "axis":2},
        # {"opName": "stack", "outName": "stack/rank3_axis-2", "varShapes":[[2,1,3], [2,1,3]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "axis":-2},
        # {"opName": "stack", "outName": "stack/rank3_axis0", "varShapes":[[2,1,3], [2,1,3]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "axis":0},
        # {"opName": "stack", "outName": "stack/rank3_axis3", "varShapes":[[2,1,3], [2,1,3]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "axis":3},
        #Note that parallel_stack doesn't support axis arg - equivalent to stack with axis=0
        # {"opName": "parallel_stack", "outName": "parallel_stack/rank0", "varShapes":[[], []], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"]},
        # {"opName": "parallel_stack", "outName": "parallel_stack/rank1", "varShapes":[[3], [3]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"]},
        # {"opName": "parallel_stack", "outName": "parallel_stack/rank2", "varShapes":[[2,3], [2,3]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"]},
        # {"opName": "parallel_stack", "outName": "parallel_stack/rank3", "varShapes":[[2,1,3], [2,1,3]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"]},
        # {"opName": "accumulate_n", "outName": "accumulate_n/rank0", "varShapes":[[], [], []], "varTypes":["float32", "float32", "float32"], "varInit":["uniform", "uniform", "uniform"]},
        # {"opName": "accumulate_n", "outName": "accumulate_n/rank1", "varShapes":[[3], [3], [3]], "varTypes":["float32", "float32", "float32"], "varInit":["uniform", "uniform", "uniform"]},
        # {"opName": "accumulate_n", "outName": "accumulate_n/rank2", "varShapes":[[2,3], [2,3], [2,3]], "varTypes":["float32", "float32", "float32"], "varInit":["uniform", "uniform", "uniform"]},
        # {"opName": "accumulate_n", "outName": "accumulate_n/rank3", "varShapes":[[2,3,4], [2,3,4], [2,3,4]], "varTypes":["float32", "float32", "float32"], "varInit":["uniform", "uniform", "uniform"]},
        # {"opName": "angle", "outName": "angle_scalar", "varShapes":[[]], "varTypes":["float32"], "varInit":["uniform"]},
        # {"opName": "angle", "outName": "angle_rank1", "varShapes":[[5]], "varTypes":["float32"], "varInit":["uniform"]},
        # {"opName": "angle", "outName": "angle_rank2", "varShapes":[[3,4]], "varTypes":["float32"], "varInit":["uniform"]},
        #TODO how to create ApproximateEqual class??
        # {"opName": "approximate_equal", "outName": "approximate_equal_scalar", "varShapes":[[],[]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "tolerance":0.1},
        # {"opName": "matmul", "outName": "matmul/rank2", "varShapes":[[3,4],[4,5]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "transpose_a":False, "transpose_b":False},
        # {"opName": "matmul", "outName": "matmul/rank2_ta", "varShapes":[[4,3],[4,5]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "transpose_a":True, "transpose_b":False},
        # {"opName": "matmul", "outName": "matmul/rank2_tb", "varShapes":[[3,4],[5,4]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "transpose_a":False, "transpose_b":True},
        # {"opName": "matmul", "outName": "matmul/rank3_batch1", "varShapes":[[1,3,4],[1,4,5]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"]},
        # {"opName": "matmul", "outName": "matmul/rank3_batch2", "varShapes":[[2,3,4],[2,4,5]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"]},
        # {"opName": "matmul", "outName": "matmul/rank3_batch2_ta", "varShapes":[[2,4,3],[2,4,5]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "transpose_a":True, "transpose_b":False},
        # {"opName": "matmul", "outName": "matmul/rank3_batch2_tb", "varShapes":[[2,3,4],[2,5,4]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "transpose_a":False, "transpose_b":True},
        # {"opName": "matmul", "outName": "matmul/rank3_batch2_ta_tb", "varShapes":[[2,4,3],[2,5,4]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "transpose_a":True, "transpose_b":True},
        # {"opName": "matmul", "outName": "matmul/rank3_batch3", "varShapes":[[3,3,4],[3,4,5]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"]},
        # {"opName": "matmul", "outName": "matmul/rank4_batch2,2", "varShapes":[[2,2,3,4],[2,2,4,5]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"]},
        # {"opName": "matmul", "outName": "matmul/rank4_batch2,2_ta", "varShapes":[[2,2,4,3],[2,2,4,5]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "transpose_a":True, "transpose_b":False},
        # {"opName": "matmul", "outName": "matmul/rank4_batch2,2_tb", "varShapes":[[2,2,3,4],[2,2,5,4]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "transpose_a":False, "transpose_b":True},
        # {"opName": "matmul", "outName": "matmul/rank4_batch2,2_ta_tb", "varShapes":[[2,2,4,3],[2,2,5,4]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "transpose_a":True, "transpose_b":True},
        # {"opName": "matmul", "outName": "matmul/rank5_batch2,2,2", "varShapes":[[2,2,2,3,4],[2,2,2,4,5]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "transpose_a":False, "transpose_b":False},
        # {"opName": "matmul", "outName": "matmul/rank5_batch2,2,2_ta_tb", "varShapes":[[2,2,2,4,3],[2,2,2,5,4]], "varTypes":["float32", "float32"], "varInit":["uniform", "uniform"], "transpose_a":True, "transpose_b":True},
        # {"opName": "matrix_diag_part", "outName": "matrix_diag_part/rank2", "varShapes":[[4,4]], "varTypes":["float32"], "varInit":["uniform"]},
        # {"opName": "matrix_diag_part", "outName": "matrix_diag_part/rank3", "varShapes":[[3,4,4]], "varTypes":["float32"], "varInit":["uniform"]},
        # {"opName": "matrix_diag_part", "outName": "matrix_diag_part/rank4", "varShapes":[[2,2,4,4]], "varTypes":["float32"], "varInit":["uniform"]},
        {"opName": "svd", "outName": "svd/rank2_3,3_noFull_noUv", "varShapes":[[3,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":False, "compute_uv":False},
        {"opName": "svd", "outName": "svd/rank2_3,3_full_noUv", "varShapes":[[3,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":True, "compute_uv":False},
        {"opName": "svd", "outName": "svd/rank2_3,3_noFull_uv", "varShapes":[[3,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":False, "compute_uv":True},
        {"opName": "svd", "outName": "svd/rank2_3,3_full_uv", "varShapes":[[3,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":True, "compute_uv":False},
        # {"opName": "svd", "outName": "svd/rank2_4,3_noFull_noUv", "varShapes":[[4,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":False, "compute_uv":False},
        # {"opName": "svd", "outName": "svd/rank2_4,3_full_noUv", "varShapes":[[4,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":True, "compute_uv":False},
        # {"opName": "svd", "outName": "svd/rank2_4,3_noFull_uv", "varShapes":[[4,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":False, "compute_uv":True},
        # {"opName": "svd", "outName": "svd/rank2_4,3_full_uv", "varShapes":[[4,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":True, "compute_uv":False},
        {"opName": "svd", "outName": "svd/rank3_2,3,3_noFull_noUv", "varShapes":[[2,3,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":False, "compute_uv":False},
        {"opName": "svd", "outName": "svd/rank3_2,3,3_full_noUv", "varShapes":[[2,3,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":True, "compute_uv":False},
        {"opName": "svd", "outName": "svd/rank3_2,3,3_noFull_uv", "varShapes":[[2,3,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":False, "compute_uv":True},
        {"opName": "svd", "outName": "svd/rank3_2,3,3_full_uv", "varShapes":[[2,3,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":True, "compute_uv":False},
        # {"opName": "svd", "outName": "svd/rank3_2,4,3_noFull_noUv", "varShapes":[[2,4,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":False, "compute_uv":False},
        # {"opName": "svd", "outName": "svd/rank3_2,4,3_full_noUv", "varShapes":[[2,4,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":True, "compute_uv":False},
        # {"opName": "svd", "outName": "svd/rank3_2,4,3_noFull_uv", "varShapes":[[2,4,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":False, "compute_uv":True},
        # {"opName": "svd", "outName": "svd/rank3_2,4,3_full_uv", "varShapes":[[2,4,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":True, "compute_uv":False},
        {"opName": "svd", "outName": "svd/rank4_2,2,3,3_noFull_noUv", "varShapes":[[2,2,3,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":False, "compute_uv":False},
        {"opName": "svd", "outName": "svd/rank4_2,2,3,3_full_noUv", "varShapes":[[2,2,3,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":True, "compute_uv":False},
        {"opName": "svd", "outName": "svd/rank4_2,2,3,3_noFull_uv", "varShapes":[[2,2,3,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":False, "compute_uv":True},
        {"opName": "svd", "outName": "svd/rank4_2,2,3,3_full_uv", "varShapes":[[2,2,3,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":True, "compute_uv":False},
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
