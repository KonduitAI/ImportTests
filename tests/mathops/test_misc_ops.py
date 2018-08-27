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
        # {"opName": "svd", "outName": "svd/rank2_3,3_noFull_noUv", "varShapes":[[3,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":False, "compute_uv":False},
        # {"opName": "svd", "outName": "svd/rank2_3,3_full_noUv", "varShapes":[[3,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":True, "compute_uv":False},
        # {"opName": "svd", "outName": "svd/rank2_3,3_noFull_uv", "varShapes":[[3,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":False, "compute_uv":True},
        # {"opName": "svd", "outName": "svd/rank2_3,3_full_uv", "varShapes":[[3,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":True, "compute_uv":False},
        # {"opName": "svd", "outName": "svd/rank2_4,3_noFull_noUv", "varShapes":[[4,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":False, "compute_uv":False},
        # {"opName": "svd", "outName": "svd/rank2_4,3_full_noUv", "varShapes":[[4,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":True, "compute_uv":False},
        # {"opName": "svd", "outName": "svd/rank2_4,3_noFull_uv", "varShapes":[[4,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":False, "compute_uv":True},
        # {"opName": "svd", "outName": "svd/rank2_4,3_full_uv", "varShapes":[[4,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":True, "compute_uv":False},
        # {"opName": "svd", "outName": "svd/rank3_2,3,3_noFull_noUv", "varShapes":[[2,3,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":False, "compute_uv":False},
        # {"opName": "svd", "outName": "svd/rank3_2,3,3_full_noUv", "varShapes":[[2,3,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":True, "compute_uv":False},
        # {"opName": "svd", "outName": "svd/rank3_2,3,3_noFull_uv", "varShapes":[[2,3,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":False, "compute_uv":True},
        # {"opName": "svd", "outName": "svd/rank3_2,3,3_full_uv", "varShapes":[[2,3,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":True, "compute_uv":False},
        # {"opName": "svd", "outName": "svd/rank3_2,4,3_noFull_noUv", "varShapes":[[2,4,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":False, "compute_uv":False},
        # {"opName": "svd", "outName": "svd/rank3_2,4,3_full_noUv", "varShapes":[[2,4,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":True, "compute_uv":False},
        # {"opName": "svd", "outName": "svd/rank3_2,4,3_noFull_uv", "varShapes":[[2,4,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":False, "compute_uv":True},
        # {"opName": "svd", "outName": "svd/rank3_2,4,3_full_uv", "varShapes":[[2,4,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":True, "compute_uv":False},
        # {"opName": "svd", "outName": "svd/rank4_2,2,3,3_noFull_noUv", "varShapes":[[2,2,3,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":False, "compute_uv":False},
        # {"opName": "svd", "outName": "svd/rank4_2,2,3,3_full_noUv", "varShapes":[[2,2,3,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":True, "compute_uv":False},
        # {"opName": "svd", "outName": "svd/rank4_2,2,3,3_noFull_uv", "varShapes":[[2,2,3,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":False, "compute_uv":True},
        # {"opName": "svd", "outName": "svd/rank4_2,2,3,3_full_uv", "varShapes":[[2,2,3,3]], "varTypes":["float32"], "varInit":["uniform"], "full_matrices":True, "compute_uv":False},

        # {"opName": "mean_squared_error", "outName": "losses/mse_rank0", "varShapes":[[],[]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"]},
        # {"opName": "mean_squared_error", "outName": "losses/mse_rank1", "varShapes":[[5],[5]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"]},
        # {"opName": "mean_squared_error", "outName": "losses/mse_rank2", "varShapes":[[3,4],[3,4]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"]},
        # {"opName": "mean_squared_error", "outName": "losses/mse_rank3", "varShapes":[[2,3,4],[2,3,4]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"]},
        # {"opName": "mean_squared_error", "outName": "losses/mse_rank1_weights_1", "varShapes":[[5],[5],[]], "varTypes":["float32","float32", "float32"], "varInit":["uniform","uniform", "uniform"]},
        # {"opName": "mean_squared_error", "outName": "losses/mse_rank2_weights_1", "varShapes":[[3,4],[3,4],[]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"]},
        # {"opName": "mean_squared_error", "outName": "losses/mse_rank2_weights_2", "varShapes":[[3,4],[3,4],[1,4]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"]},
        # {"opName": "mean_squared_error", "outName": "losses/mse_rank2_weights_3", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"]},
        # {"opName": "mean_squared_error", "outName": "losses/mse_rank3_weights_1", "varShapes":[[2,3,4],[2,3,4],[]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"]},
        # {"opName": "mean_squared_error", "outName": "losses/mse_rank3_weights_2", "varShapes":[[2,3,4],[2,3,4],[1,1,4]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"]},
        # {"opName": "mean_squared_error", "outName": "losses/mse_rank3_weights_3", "varShapes":[[2,3,4],[2,3,4],[2,1,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"]},
        # {"opName": "mean_squared_error", "outName": "losses/mse_rank3_weights_4", "varShapes":[[2,3,4],[2,3,4],[2,3,4]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"]}
        # {"opName": "absolute_difference", "outName": "losses/absdiff_rank0", "varShapes":[[],[]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"]},
        # {"opName": "absolute_difference", "outName": "losses/absdiff_rank1", "varShapes":[[5],[5]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"]},
        # {"opName": "absolute_difference", "outName": "losses/absdiff_rank2", "varShapes":[[3,4],[3,4]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"]},
        # {"opName": "absolute_difference", "outName": "losses/absdiff_rank3", "varShapes":[[2,3,4],[2,3,4]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"]},
        # {"opName": "absolute_difference", "outName": "losses/absdiff_rank0_weights", "varShapes":[[],[],[]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"]},
        # {"opName": "absolute_difference", "outName": "losses/absdiff_rank1_weights_1", "varShapes":[[5],[5],[]], "varTypes":["float32","float32", "float32"], "varInit":["uniform","uniform", "uniform"]},
        # {"opName": "absolute_difference", "outName": "losses/absdiff_rank2_weights_1", "varShapes":[[3,4],[3,4],[]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"]},
        # {"opName": "absolute_difference", "outName": "losses/absdiff_rank2_weights_2", "varShapes":[[3,4],[3,4],[1,4]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"]},
        # {"opName": "absolute_difference", "outName": "losses/absdiff_rank3_weights_1", "varShapes":[[2,3,4],[2,3,4],[]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"]},
        # {"opName": "absolute_difference", "outName": "losses/absdiff_rank3_weights_2", "varShapes":[[2,3,4],[2,3,4],[2,1,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"]},

        #{"opName": "cosine_distance", "outName": "losses/cosine_diff_rank0", "varShapes":[[],[]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"], "axis":None},     #Cosine doesn't like rank 0 input, it seems...
        # {"opName": "cosine_distance", "outName": "losses/cosine_diff_rank1", "varShapes":[[5],[5]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"], "axis":0},
        # {"opName": "cosine_distance", "outName": "losses/cosine_diff_rank2_axis0_SUM", "varShapes":[[3,4],[3,4],[1,4]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform_sparse"], "reduction":tf.losses.Reduction.SUM, "axis":0},
        # {"opName": "cosine_distance", "outName": "losses/cosine_diff_rank2_axis1_NONE", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform_sparse"], "reduction":tf.losses.Reduction.NONE, "axis":1},
        # {"opName": "cosine_distance", "outName": "losses/cosine_diff_rank2_axis1_SUM", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform_sparse"], "reduction":tf.losses.Reduction.SUM, "axis":1},
        # {"opName": "cosine_distance", "outName": "losses/cosine_diff_rank2_axis1_MEAN", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform_sparse"], "reduction":tf.losses.Reduction.MEAN, "axis":1},
        # {"opName": "cosine_distance", "outName": "losses/cosine_diff_rank2_axis1_SUM_OVER_BATCH_SIZE", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform_sparse"], "reduction":tf.losses.Reduction.SUM_OVER_BATCH_SIZE, "axis":1},
        # {"opName": "cosine_distance", "outName": "losses/cosine_diff_rank2_axis1_SUM_BY_NONZERO_WEIGHTS", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform_sparse"], "reduction":tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS, "axis":1},
        # {"opName": "cosine_distance", "outName": "losses/cosine_diff_rank3", "varShapes":[[2,3,4],[2,3,4]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"], "axis":0},
        # {"opName": "cosine_distance", "outName": "losses/cosine_diff_rank3_axis1", "varShapes":[[2,3,4],[2,3,4]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"], "axis":1},
        # {"opName": "cosine_distance", "outName": "losses/cosine_diff_rank3_axis2", "varShapes":[[2,3,4],[2,3,4]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"], "axis":2},
        # {"opName": "cosine_distance", "outName": "losses/cosine_diff_rank3_weights0", "varShapes":[[2,3,4],[2,3,4],[1,1,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"], "axis":0},  #Can't have weights [2,1,1]? Maybe weights must match post reduce...
        # {"opName": "cosine_distance", "outName": "losses/cosine_diff_rank3_weights1", "varShapes":[[2,3,4],[2,3,4],[1,3,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"], "axis":0},
        # {"opName": "cosine_distance", "outName": "losses/cosine_diff_rank3_weights2", "varShapes":[[2,3,4],[2,3,4],[1,1,4]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"], "axis":0},
        # {"opName": "cosine_distance", "outName": "losses/cosine_diff_rank3_weightsAll", "varShapes":[[2,3,4],[2,3,4],[1,3,4]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"], "axis":0},

        #Hinge: need bernoulli (0 or 1) labels, and 0 centered predictions (<0 negative, > 0 positive)
        # {"opName": "hinge_loss", "outName": "losses/hinge_rank1", "varShapes":[[5],[5]], "varTypes":["float32","float32"], "varInit":["bernoulli","uniform_m1_1"]},
        # {"opName": "hinge_loss", "outName": "losses/hinge_rank2_NONE", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["bernoulli","uniform_m1_1","uniform_sparse"], "reduction":tf.losses.Reduction.NONE},
        # {"opName": "hinge_loss", "outName": "losses/hinge_rank2_SUM", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["bernoulli","uniform_m1_1","uniform_sparse"], "reduction":tf.losses.Reduction.SUM},
        # {"opName": "hinge_loss", "outName": "losses/hinge_rank2_MEAN", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["bernoulli","uniform_m1_1","uniform_sparse"], "reduction":tf.losses.Reduction.MEAN},
        # {"opName": "hinge_loss", "outName": "losses/hinge_rank2_SUM_OVER_BATCH_SIZE", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["bernoulli","uniform_m1_1","uniform_sparse"], "reduction":tf.losses.Reduction.SUM_OVER_BATCH_SIZE},
        # {"opName": "hinge_loss", "outName": "losses/hinge_rank2_SUM_BY_NONZERO_WEIGHTS", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["bernoulli","uniform_m1_1","uniform_sparse"], "reduction":tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS},
        # {"opName": "hinge_loss", "outName": "losses/hinge_rank3", "varShapes":[[2,3,4],[2,3,4]], "varTypes":["float32","float32"], "varInit":["bernoulli","uniform_m1_1"]},
        # {"opName": "hinge_loss", "outName": "losses/hinge_rank3_weights0", "varShapes":[[2,3,4],[2,3,4],[2,1,1]], "varTypes":["float32","float32","float32"], "varInit":["bernoulli","uniform_m1_1","uniform"]},
        # {"opName": "hinge_loss", "outName": "losses/hinge_rank3_weights1", "varShapes":[[2,3,4],[2,3,4],[1,3,1]], "varTypes":["float32","float32","float32"], "varInit":["bernoulli","uniform_m1_1","uniform"]},
        # {"opName": "hinge_loss", "outName": "losses/hinge_rank3_weights2", "varShapes":[[2,3,4],[2,3,4],[1,1,4]], "varTypes":["float32","float32","float32"], "varInit":["bernoulli","uniform_m1_1","uniform"]},
        # {"opName": "hinge_loss", "outName": "losses/hinge_rank3_weightsAll", "varShapes":[[2,3,4],[2,3,4],[2,3,4]], "varTypes":["float32","float32","float32"], "varInit":["bernoulli","uniform_m1_1","uniform"]},

        # {"opName": "huber_loss", "outName": "losses/huber_diff_rank0", "varShapes":[[],[]], "varTypes":["float32","float32"], "varInit":["stdnormal","stdnormal"]},
        # {"opName": "huber_loss", "outName": "losses/huber_diff_rank0", "varShapes":[[],[]], "varTypes":["float32","float32"], "varInit":["stdnormal","stdnormal"]},
        # {"opName": "huber_loss", "outName": "losses/huber_diff_rank1", "varShapes":[[5],[5]], "varTypes":["float32","float32"], "varInit":["stdnormal","stdnormal"]},
        # {"opName": "huber_loss", "outName": "losses/huber_diff_rank1_d05", "varShapes":[[5],[5]], "varTypes":["float32","float32"], "varInit":["stdnormal","stdnormal"],"delta":0.5},
        # {"opName": "huber_loss", "outName": "losses/huber_diff_rank1_d2", "varShapes":[[5],[5]], "varTypes":["float32","float32"], "varInit":["stdnormal","stdnormal"],"delta":2.0},
        # {"opName": "huber_loss", "outName": "losses/huber_diff_rank2_axis0_SUM", "varShapes":[[3,4],[3,4],[1,4]], "varTypes":["float32","float32","float32"], "varInit":["stdnormal","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.SUM},
        # {"opName": "huber_loss", "outName": "losses/huber_diff_rank2_axis1_NONE", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["stdnormal","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.NONE},
        # {"opName": "huber_loss", "outName": "losses/huber_diff_rank2_axis1_SUM", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["stdnormal","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.SUM},
        # {"opName": "huber_loss", "outName": "losses/huber_diff_rank2_axis1_MEAN", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["stdnormal","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.MEAN},
        # {"opName": "huber_loss", "outName": "losses/huber_diff_rank2_axis1_SUM_OVER_BATCH_SIZE", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["stdnormal","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.SUM_OVER_BATCH_SIZE},
        # {"opName": "huber_loss", "outName": "losses/huber_diff_rank2_axis1_SUM_BY_NONZERO_WEIGHTS", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["stdnormal","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS},
        # {"opName": "huber_loss", "outName": "losses/huber_diff_rank3", "varShapes":[[2,3,4],[2,3,4]], "varTypes":["float32","float32"], "varInit":["stdnormal","stdnormal"]},
        # {"opName": "huber_loss", "outName": "losses/huber_diff_rank3_axis1", "varShapes":[[2,3,4],[2,3,4]], "varTypes":["float32","float32"], "varInit":["stdnormal","stdnormal"]},
        # {"opName": "huber_loss", "outName": "losses/huber_diff_rank3_axis2", "varShapes":[[2,3,4],[2,3,4]], "varTypes":["float32","float32"], "varInit":["stdnormal","stdnormal"]},
        # {"opName": "huber_loss", "outName": "losses/huber_diff_rank3_weights0", "varShapes":[[2,3,4],[2,3,4],[1,1,1]], "varTypes":["float32","float32","float32"], "varInit":["stdnormal","stdnormal","uniform"]},
        # {"opName": "huber_loss", "outName": "losses/huber_diff_rank3_weights1", "varShapes":[[2,3,4],[2,3,4],[1,3,1]], "varTypes":["float32","float32","float32"], "varInit":["stdnormal","stdnormal","uniform"]},
        # {"opName": "huber_loss", "outName": "losses/huber_diff_rank3_weights2", "varShapes":[[2,3,4],[2,3,4],[1,1,4]], "varTypes":["float32","float32","float32"], "varInit":["stdnormal","stdnormal","uniform"]},
        # {"opName": "huber_loss", "outName": "losses/huber_diff_rank3_weightsAll", "varShapes":[[2,3,4],[2,3,4],[1,3,4]], "varTypes":["float32","float32","float32"], "varInit":["stdnormal","stdnormal","uniform"]},

        # {"opName": "log_loss", "outName": "losses/log_loss_rank0", "varShapes":[[],[]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"]},
        # {"opName": "log_loss", "outName": "losses/log_loss_rank0", "varShapes":[[],[]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"]},
        # {"opName": "log_loss", "outName": "losses/log_loss_rank1", "varShapes":[[5],[5]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"]},
        # {"opName": "log_loss", "outName": "losses/log_loss_rank1_eps01", "varShapes":[[5],[5]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"],"epsilon":0.1},
        # {"opName": "log_loss", "outName": "losses/log_loss_rank2_axis0_SUM", "varShapes":[[3,4],[3,4],[1,4]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform_sparse"], "reduction":tf.losses.Reduction.SUM},
        # {"opName": "log_loss", "outName": "losses/log_loss_rank2_axis1_NONE", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform_sparse"], "reduction":tf.losses.Reduction.NONE},
        # {"opName": "log_loss", "outName": "losses/log_loss_rank2_axis1_SUM", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform_sparse"], "reduction":tf.losses.Reduction.SUM},
        # {"opName": "log_loss", "outName": "losses/log_loss_rank2_axis1_MEAN", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform_sparse"], "reduction":tf.losses.Reduction.MEAN},
        # {"opName": "log_loss", "outName": "losses/log_loss_rank2_axis1_SUM_OVER_BATCH_SIZE", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform_sparse"], "reduction":tf.losses.Reduction.SUM_OVER_BATCH_SIZE},
        # {"opName": "log_loss", "outName": "losses/log_loss_rank2_axis1_SUM_BY_NONZERO_WEIGHTS", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform_sparse"], "reduction":tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS},
        # {"opName": "log_loss", "outName": "losses/log_loss_rank3", "varShapes":[[2,3,4],[2,3,4]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"]},
        # {"opName": "log_loss", "outName": "losses/log_loss_rank3_axis1", "varShapes":[[2,3,4],[2,3,4]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"]},
        # {"opName": "log_loss", "outName": "losses/log_loss_rank3_axis2", "varShapes":[[2,3,4],[2,3,4]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"]},
        # {"opName": "log_loss", "outName": "losses/log_loss_rank3_weights0", "varShapes":[[2,3,4],[2,3,4],[1,1,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"]},
        # {"opName": "log_loss", "outName": "losses/log_loss_rank3_weights1", "varShapes":[[2,3,4],[2,3,4],[1,3,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"]},
        # {"opName": "log_loss", "outName": "losses/log_loss_rank3_weights2", "varShapes":[[2,3,4],[2,3,4],[1,1,4]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"]},
        # {"opName": "log_loss", "outName": "losses/log_loss_rank3_weightsAll", "varShapes":[[2,3,4],[2,3,4],[1,3,4]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"]},

        #sigmoid_cross_entropy: seems to only support [batch_size, num_classes] shapes?
        # {"opName": "sigmoid_cross_entropy", "outName": "losses/sigmoid_ce", "varShapes":[[3,4],[3,4]], "varTypes":["float32","float32"], "varInit":["uniform","stdnormal"]},
        # {"opName": "sigmoid_cross_entropy", "outName": "losses/sigmoid_ce_smooth01", "varShapes":[[3,4],[3,4]], "varTypes":["float32","float32"], "varInit":["uniform","stdnormal"],"label_smoothing":0.1},
        # {"opName": "sigmoid_cross_entropy", "outName": "losses/sigmoid_ce_smooth05", "varShapes":[[3,4],[3,4]], "varTypes":["float32","float32"], "varInit":["uniform","stdnormal"],"label_smoothing":0.5},
        # {"opName": "sigmoid_cross_entropy", "outName": "losses/sigmoid_ce_NONE", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.NONE},
        # {"opName": "sigmoid_cross_entropy", "outName": "losses/sigmoid_ce_SUM", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.SUM},
        # {"opName": "sigmoid_cross_entropy", "outName": "losses/sigmoid_ce_MEAN", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.MEAN},
        # {"opName": "sigmoid_cross_entropy", "outName": "losses/sigmoid_ce_SUM_OVER_BATCH_SIZE", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.SUM_OVER_BATCH_SIZE},
        # {"opName": "sigmoid_cross_entropy", "outName": "losses/sigmoid_ce_SUM_BY_NONZERO_WEIGHTS", "varShapes":[[3,4],[3,4],[3,1]], "varTypes":["float32","float32","float32"], "varInit":["uniform","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS},

        # {"opName": "softmax_cross_entropy", "outName": "losses/softmax_ce", "varShapes":[[10,4],[10,4]], "varTypes":["float32","float32"], "varInit":["onehot","stdnormal"]},
        # {"opName": "softmax_cross_entropy", "outName": "losses/softmax_ce_smooth01", "varShapes":[[10,4],[10,4]], "varTypes":["float32","float32"], "varInit":["onehot","stdnormal"],"label_smoothing":0.1},
        # {"opName": "softmax_cross_entropy", "outName": "losses/softmax_ce_smooth05", "varShapes":[[10,4],[10,4]], "varTypes":["float32","float32"], "varInit":["onehot","stdnormal"],"label_smoothing":0.5},
        # {"opName": "softmax_cross_entropy", "outName": "losses/softmax_ce_NONE", "varShapes":[[10,4],[10,4],[10]], "varTypes":["float32","float32","float32"], "varInit":["onehot","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.NONE},
        # {"opName": "softmax_cross_entropy", "outName": "losses/softmax_ce_SUM", "varShapes":[[10,4],[10,4],[10]], "varTypes":["float32","float32","float32"], "varInit":["onehot","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.SUM},
        # {"opName": "softmax_cross_entropy", "outName": "losses/softmax_ce_MEAN", "varShapes":[[10,4],[10,4],[10]], "varTypes":["float32","float32","float32"], "varInit":["onehot","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.MEAN},
        # {"opName": "softmax_cross_entropy", "outName": "losses/softmax_ce_SUM_OVER_BATCH_SIZE", "varShapes":[[10,4],[10,4],[10]], "varTypes":["float32","float32","float32"], "varInit":["onehot","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.SUM_OVER_BATCH_SIZE},
        # {"opName": "softmax_cross_entropy", "outName": "losses/softmax_ce_SUM_BY_NONZERO_WEIGHTS", "varShapes":[[10,4],[10,4],[10]], "varTypes":["float32","float32","float32"], "varInit":["onehot","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS},

        # {"opName": "sparse_softmax_cross_entropy", "outName": "losses/sparse_softmax_ce", "varShapes":[[10],[10,5]], "varTypes":["int32","float32"], "varInit":["uniform_int5","stdnormal"]},
        # {"opName": "sparse_softmax_cross_entropy", "outName": "losses/sparse_softmax_ce_NONE", "varShapes":[[10],[10,5],[10,1]], "varTypes":["int32","float32","float32"], "varInit":["uniform_int5","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.NONE},
        # {"opName": "sparse_softmax_cross_entropy", "outName": "losses/sparse_softmax_ce_SUM", "varShapes":[[10],[10,5],[10,1]], "varTypes":["int32","float32","float32"], "varInit":["uniform_int5","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.SUM},
        # {"opName": "sparse_softmax_cross_entropy", "outName": "losses/sparse_softmax_ce_MEAN", "varShapes":[[10],[10,5],[10,1]], "varTypes":["int32","float32","float32"], "varInit":["uniform_int5","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.MEAN},
        # {"opName": "sparse_softmax_cross_entropy", "outName": "losses/sparse_softmax_ce_SUM_OVER_BATCH_SIZE", "varShapes":[[10],[10,5],[10]], "varTypes":["int32","float32","float32"], "varInit":["uniform_int5","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.SUM_OVER_BATCH_SIZE},
        # {"opName": "sparse_softmax_cross_entropy", "outName": "losses/sparse_softmax_ce_SUM_BY_NONZERO_WEIGHTS", "varShapes":[[10],[10,5],[10]], "varTypes":["int32","float32","float32"], "varInit":["uniform_int5","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS},
        # {"opName": "sparse_softmax_cross_entropy", "outName": "losses/sparse_softmax_ce_SUM_BY_NONZERO_WEIGHTS_1", "varShapes":[[10],[10,5],[1]], "varTypes":["int32","float32","float32"], "varInit":["uniform_int5","stdnormal","uniform_sparse"], "reduction":tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS},

        # {"opName":"l2_loss", "outName":"losses/l2_loss_rank0", "varShapes":[[]], "varTypes":["float32"], "varInit":["uniform"]},
        # {"opName":"l2_loss", "outName":"losses/l2_loss_rank1", "varShapes":[[10]], "varTypes":["float32"], "varInit":["uniform"]},
        # {"opName":"l2_loss", "outName":"losses/l2_loss_rank2", "varShapes":[[3,4]], "varTypes":["float32"], "varInit":["uniform"]},
        # {"opName":"l2_loss", "outName":"losses/l2_loss_rank3", "varShapes":[[2,3,4]], "varTypes":["float32"], "varInit":["uniform"]},
        # {"opName":"l2_loss", "outName":"losses/l2_loss_rank4", "varShapes":[[2,3,4,5]], "varTypes":["float32"], "varInit":["uniform"]}

        #tf.nn.conv1d
        #CNN 1D layers: value, filters
        #value: NCW: [batch, channels, width], NWC: [batch, width, channels]
        #filters are [kernel, inChannels, outChannels] for both
        #Can't run the ncw tests: "UnimplementedError (see above for traceback): Generic conv implementation only supports NHWC tensor format for now" :/
        # {"opName":"nn_cnn1d", "outName":"cnn1d_nn/ncw_b1_k2_s1_SAME", "varShapes":[[1, 2, 5], [5, 2, 3]], "varTypes":["float32","float32"], "stride":1, "padding":"SAME", "data_format":"NCW"},
        # {"opName":"nn_cnn1d", "outName":"cnn1d_nn/ncw_b2_k2_s1_SAME", "varShapes":[[2, 2, 5], [5, 2, 3]], "varTypes":["float32","float32"], "stride":1, "padding":"SAME", "data_format":"NCW"},
        # {"opName":"nn_cnn1d", "outName":"cnn1d_nn/ncw_b2_k2_s1_VALID", "varShapes":[[2, 2, 5], [5, 2, 3]], "varTypes":["float32","float32"], "stride":1, "padding":"VALID", "data_format":"NCW"},
        # {"opName":"nn_cnn1d", "outName":"cnn1d_nn/ncw_b1_k2_s2_SAME", "varShapes":[[1, 2, 5], [5, 2, 3]], "varTypes":["float32","float32"], "stride":2, "padding":"SAME", "data_format":"NCW"},
        # {"opName":"nn_cnn1d", "outName":"cnn1d_nn/nwc_b1_k2_s1_SAME", "varShapes":[[1, 5, 2], [5, 2, 3]], "varTypes":["float32","float32"], "stride":1, "padding":"SAME", "data_format":"NWC"},
        # {"opName":"nn_cnn1d", "outName":"cnn1d_nn/nwc_b2_k2_s1_SAME", "varShapes":[[2, 5, 2], [5, 2, 3]], "varTypes":["float32","float32"], "stride":1, "padding":"SAME", "data_format":"NWC"},
        # {"opName":"nn_cnn1d", "outName":"cnn1d_nn/nwc_b2_k2_s1_VALID", "varShapes":[[2, 5, 2], [5, 2, 3]], "varTypes":["float32","float32"], "stride":1, "padding":"VALID", "data_format":"NWC"},
        # {"opName":"nn_cnn1d", "outName":"cnn1d_nn/nwc_b1_k2_s2_SAME", "varShapes":[[1, 5, 2], [5, 2, 3]], "varTypes":["float32","float32"], "stride":2, "padding":"SAME", "data_format":"NWC"},

        #tf.layers.conv1d
        #Note that the tf.layers version seems to add the variables directly - you don't provide the kernel params as a variable...
        #Also can't run channels_first here: "Generic conv implementation only supports NHWC tensor format for now." :/
        # {"opName":"layers_cnn1d", "outName":"cnn1d_layers/channels_first_b1_k2_s1_d1_SAME", "varShapes":[[1, 2, 5]], "varTypes":["float32","float32"], "filters":2, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_first", "dilation_rate":1},
        # {"opName":"layers_cnn1d", "outName":"cnn1d_layers/channels_first_b1_k3_s1_d1_SAME", "varShapes":[[1, 2, 5]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_first", "dilation_rate":1},
        # {"opName":"layers_cnn1d", "outName":"cnn1d_layers/channels_first_b2_k2_s1_d2_SAME", "varShapes":[[2, 2, 5]], "varTypes":["float32","float32"], "filters":4, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_first", "dilation_rate":2},
        # {"opName":"layers_cnn1d", "outName":"cnn1d_layers/channels_first_b2_k2_s1_d1_VALID", "varShapes":[[2, 2, 5]], "varTypes":["float32","float32"], "filters":1, "kernel_size":2, "strides":1, "padding":"VALID", "data_format":"channels_first", "dilation_rate":1},
        # {"opName":"layers_cnn1d", "outName":"cnn1d_layers/channels_first_b1_k2_s2_d1_SAME", "varShapes":[[1, 2, 5]], "varTypes":["float32","float32"], "filters":2, "kernel_size":2, "strides":2, "padding":"SAME", "data_format":"channels_first", "dilation_rate":1},
        # {"opName":"layers_cnn1d", "outName":"cnn1d_layers/channels_last_b1_k2_s1_d1_SAME", "varShapes":[[1, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1},
        # {"opName":"layers_cnn1d", "outName":"cnn1d_layers/channels_last_b1_k2_s1_d2_SAME", "varShapes":[[1, 5, 2]], "varTypes":["float32","float32"], "filters":2, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":2},
        # {"opName":"layers_cnn1d", "outName":"cnn1d_layers/channels_last_b2_k2_s1_SAME", "varShapes":[[2, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1},
        # {"opName":"layers_cnn1d", "outName":"cnn1d_layers/channels_last_b2_k2_s1_VALID", "varShapes":[[2, 5, 2]], "varTypes":["float32","float32"], "filters":2, "kernel_size":2, "strides":1, "padding":"VALID", "data_format":"channels_last", "dilation_rate":1},
        # {"opName":"layers_cnn1d", "outName":"cnn1d_layers/channels_last_b1_k2_s2_SAME", "varShapes":[[1, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":2, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1},
        # {"opName":"layers_cnn1d", "outName":"cnn1d_layers/channels_last_b1_k2_s1_d1_SAME_sigmoid", "varShapes":[[1, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1, "activation":tf.nn.relu},
        # {"opName":"layers_cnn1d", "outName":"cnn1d_layers/channels_last_b1_k2_s1_d1_SAME_sigmoid", "varShapes":[[1, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1, "activation":tf.nn.sigmoid},
        # {"opName":"layers_cnn1d", "outName":"cnn1d_layers/channels_last_b1_k2_s1_d1_SAME_elu", "varShapes":[[1, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1, "activation":tf.nn.elu},
        # {"opName":"layers_cnn1d", "outName":"cnn1d_layers/channels_last_b1_k2_s1_d1_SAME_relu6", "varShapes":[[1, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1, "activation":tf.nn.relu6},
        # {"opName":"layers_cnn1d", "outName":"cnn1d_layers/channels_last_b1_k2_s1_d1_SAME_relu6", "varShapes":[[1, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1, "activation":tf.nn.selu},
        # {"opName":"layers_cnn1d", "outName":"cnn1d_layers/channels_last_b1_k2_s1_d1_SAME_crelu", "varShapes":[[1, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1, "activation":tf.nn.crelu},
        # {"opName":"layers_cnn1d", "outName":"cnn1d_layers/channels_last_b2_k2_s1_SAME_regularizers", "varShapes":[[2, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1,
        #     "kernel_regularizer":tf.contrib.layers.l2_regularizer(scale=0.1), "bias_regularizer":tf.contrib.layers.l1_regularizer(scale=0.2), "activity_regularizer":tf.contrib.layers.l1_l2_regularizer(scale_l1=0.1,scale_l2=0.2)},
        # {"opName":"layers_cnn1d", "outName":"cnn1d_layers/channels_last_b2_k2_s1_SAME_constraints1", "varShapes":[[2, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1,
        #     "kernel_constraint":tf.keras.constraints.MaxNorm(max_value=2), "bias_constraint":tf.keras.constraints.NonNeg()},
        # {"opName":"layers_cnn1d", "outName":"cnn1d_layers/channels_last_b2_k2_s1_SAME_constraints2", "varShapes":[[2, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1,
        #     "kernel_constraint":tf.keras.constraints.MinMaxNorm(min_value=1, max_value=2)},
        # {"opName":"layers_cnn1d", "outName":"cnn1d_layers/channels_last_b2_k2_s1_SAME_constraints3", "varShapes":[[2, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1,
        #     "kernel_constraint":tf.keras.constraints.UnitNorm()},
        #TODO TF constraints don't appear to get saved with the model...

        # {"opName":"layers_sepconv1d", "outName":"sepconv1d_layers/channels_last_b1_k2_s1_d1_SAME_dm1", "varShapes":[[1, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1, "depth_multiplier":1},
        # {"opName":"layers_sepconv1d", "outName":"sepconv1d_layers/channels_last_b2_k2_s1_d1_SAME_dm2", "varShapes":[[2, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1, "depth_multiplier":2},
        # {"opName":"layers_sepconv1d", "outName":"sepconv1d_layers/channels_last_b2_k2_s2_d1_SAME_dm2", "varShapes":[[2, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":2, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1, "depth_multiplier":2},
        # {"opName":"layers_sepconv1d", "outName":"sepconv1d_layers/channels_last_b2_k2_s1_d2_SAME_dm2", "varShapes":[[2, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":2, "depth_multiplier":2},
        # {"opName":"layers_sepconv1d", "outName":"sepconv1d_layers/channels_last_b2_k2_s1_d1_SAME_dm1_sigm", "varShapes":[[2, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1, "depth_multiplier":1, "activation":tf.nn.tanh},
        # {"opName":"layers_sepconv1d", "outName":"sepconv1d_layers/channels_last_b2_k2_s1_d1_SAME_dm2_sigm_nobias", "varShapes":[[2, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1, "depth_multiplier":2, "activation":tf.nn.tanh, "use_bias":False},
        # {"opName":"layers_sepconv1d", "outName":"sepconv1d_layers/channels_last_b1_k2_s1_d1_VALID_dm1", "varShapes":[[1, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"VALID", "data_format":"channels_last", "dilation_rate":1, "depth_multiplier":1},
        # {"opName":"layers_sepconv1d", "outName":"sepconv1d_layers/channels_last_b2_k2_s1_d1_VALID_dm2", "varShapes":[[2, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"VALID", "data_format":"channels_last", "dilation_rate":1, "depth_multiplier":2},
        # {"opName":"layers_sepconv1d", "outName":"sepconv1d_layers/channels_last_b2_k2_s2_d1_VALID_dm2", "varShapes":[[2, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":2, "padding":"VALID", "data_format":"channels_last", "dilation_rate":1, "depth_multiplier":2},
        # {"opName":"layers_sepconv1d", "outName":"sepconv1d_layers/channels_last_b2_k2_s1_d2_VALID_dm2", "varShapes":[[2, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"VALID", "data_format":"channels_last", "dilation_rate":2, "depth_multiplier":2},
        # {"opName":"layers_sepconv1d", "outName":"sepconv1d_layers/channels_last_b2_k2_s1_d1_VALID_dm1_sigm", "varShapes":[[2, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"VALID", "data_format":"channels_last", "dilation_rate":1, "depth_multiplier":1, "activation":tf.nn.tanh},
        # {"opName":"layers_sepconv1d", "outName":"sepconv1d_layers/channels_last_b2_k2_s1_d1_VALID_dm2_sigm_nobias", "varShapes":[[2, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"VALID", "data_format":"channels_last", "dilation_rate":1, "depth_multiplier":2, "activation":tf.nn.tanh, "use_bias":False},


        #Max/avg pool 1d:
        #channels_first:    "Default MaxPoolingOp only supports NHWC on device type CPU"        :/
        # {"opName":"max_pooling1d", "outName":"max_pooling1d/channels_first_b1_k2_s1_SAME", "varShapes":[[1, 2, 5]], "varTypes":["float32"], "pooling_size":2, "stride":1, "padding":"SAME", "data_format":"channels_first"},
        # {"opName":"max_pooling1d", "outName":"max_pooling1d/channels_first_b1_k3_s1_SAME", "varShapes":[[1, 2, 5]], "varTypes":["float32"], "pooling_size":3, "stride":1, "padding":"SAME", "data_format":"channels_first"},
        # {"opName":"max_pooling1d", "outName":"max_pooling1d/channels_first_b2_k2_s1_SAME", "varShapes":[[2, 2, 5]], "varTypes":["float32"], "pooling_size":2, "stride":1, "padding":"SAME", "data_format":"channels_first"},
        # {"opName":"max_pooling1d", "outName":"max_pooling1d/channels_first_b2_k2_s1_VALID", "varShapes":[[2, 2, 5]], "varTypes":["float32"], "pooling_size":2, "stride":1, "padding":"VALID", "data_format":"channels_first"},
        # {"opName":"max_pooling1d", "outName":"max_pooling1d/channels_first_b1_k2_s2_SAME", "varShapes":[[1, 2, 5]], "varTypes":["float32"], "pooling_size":2, "stride":2, "padding":"SAME", "data_format":"channels_first"},
        # {"opName":"max_pooling1d", "outName":"max_pooling1d/channels_last_b1_k2_s1_SAME", "varShapes":[[1, 5, 2]], "varTypes":["float32"], "pooling_size":2, "stride":1, "padding":"SAME", "data_format":"channels_last"},
        # {"opName":"max_pooling1d", "outName":"max_pooling1d/channels_last_b2_k2_s1_SAME", "varShapes":[[2, 5, 2]], "varTypes":["float32"], "pooling_size":2, "stride":1, "padding":"SAME", "data_format":"channels_last"},
        # {"opName":"max_pooling1d", "outName":"max_pooling1d/channels_last_b2_k2_s1_VALID", "varShapes":[[2, 5, 2]], "varTypes":["float32"], "pooling_size":2, "stride":1, "padding":"VALID", "data_format":"channels_last"},
        # {"opName":"max_pooling1d", "outName":"max_pooling1d/channels_last_b1_k2_s2_SAME", "varShapes":[[1, 5, 2]], "varTypes":["float32"], "pooling_size":2, "stride":2, "padding":"SAME", "data_format":"channels_last"},

        #Default AvgPoolingOp only supports NHWC on device type CPU
        # {"opName":"avg_pooling1d", "outName":"avg_pooling1d/channels_first_b1_k2_s1_SAME", "varShapes":[[1, 2, 5]], "varTypes":["float32"], "pooling_size":2, "stride":1, "padding":"SAME", "data_format":"channels_first"},
        # {"opName":"avg_pooling1d", "outName":"avg_pooling1d/channels_first_b1_k2_s1_SAME", "varShapes":[[1, 2, 5]], "varTypes":["float32"], "pooling_size":3, "stride":1, "padding":"SAME", "data_format":"channels_first"},
        # {"opName":"avg_pooling1d", "outName":"avg_pooling1d/channels_first_b2_k2_s1_SAME", "varShapes":[[2, 2, 5]], "varTypes":["float32"], "pooling_size":2, "stride":1, "padding":"SAME", "data_format":"channels_first"},
        # {"opName":"avg_pooling1d", "outName":"avg_pooling1d/channels_first_b2_k2_s1_VALID", "varShapes":[[2, 2, 5]], "varTypes":["float32"], "pooling_size":2, "stride":1, "padding":"VALID", "data_format":"channels_first"},
        # {"opName":"avg_pooling1d", "outName":"avg_pooling1d/channels_first_b1_k2_s2_SAME", "varShapes":[[1, 2, 5]], "varTypes":["float32"], "pooling_size":2, "stride":2, "padding":"SAME", "data_format":"channels_first"},
        # {"opName":"avg_pooling1d", "outName":"avg_pooling1d/channels_last_b1_k2_s1_SAME", "varShapes":[[1, 5, 2]], "varTypes":["float32"], "pooling_size":2, "stride":1, "padding":"SAME", "data_format":"channels_last"},
        # {"opName":"avg_pooling1d", "outName":"avg_pooling1d/channels_last_b2_k2_s1_SAME", "varShapes":[[2, 5, 2]], "varTypes":["float32"], "pooling_size":2, "stride":1, "padding":"SAME", "data_format":"channels_last"},
        # {"opName":"avg_pooling1d", "outName":"avg_pooling1d/channels_last_b2_k2_s1_VALID", "varShapes":[[2, 5, 2]], "varTypes":["float32"], "pooling_size":2, "stride":1, "padding":"VALID", "data_format":"channels_last"},
        # {"opName":"avg_pooling1d", "outName":"avg_pooling1d/channels_last_b1_k2_s2_SAME", "varShapes":[[1, 5, 2]], "varTypes":["float32"], "pooling_size":2, "stride":2, "padding":"SAME", "data_format":"channels_last"},

        # {"opName":"dense", "outName":"dense/dense5", "varShapes":[[5,4]], "varTypes":["float32"], "units":5, "activation":None, "use_bias":True, "kernel_regularizer":None, "bias_regularizer":None},
        # {"opName":"dense", "outName":"dense/dense5_sigmoid_nobias", "varShapes":[[5,4]], "varTypes":["float32"], "units":5, "activation":tf.nn.sigmoid, "use_bias":False, "kernel_regularizer":None, "bias_regularizer":None},
        # {"opName":"dense", "outName":"dense/dense5_tanh_regularizer", "varShapes":[[5,4]], "varTypes":["float32"], "units":5, "activation":tf.nn.tanh, "use_bias":True, "kernel_regularizer":tf.contrib.layers.l2_regularizer(scale=0.1), "bias_regularizer":None},
        # {"opName":"flatten", "outName":"flatten/rank2", "varShapes":[[3,4]], "varTypes":["float32"]},
        # {"opName":"flatten", "outName":"flatten/rank3", "varShapes":[[2,3,4]], "varTypes":["float32"]},
        # {"opName":"flatten", "outName":"flatten/rank4", "varShapes":[[2,3,2,4]], "varTypes":["float32"]},
        # {"opName":"flatten", "outName":"flatten/rank5", "varShapes":[[2,3,2,4,2]], "varTypes":["float32"]},

        # NHWC format: kernel format is [kH,kW,cIn,cOut]
        #Also, strides and dilation are 4d for some reason, strides should be [1, sH, sW, 1]
        # {"opName":"nn_conv2d", "outName":"cnn2d_nn/nhwc_b1_k2_s1_d1_SAME", "varShapes":[[1, 5, 5, 2], [2, 2, 2, 3]], "varTypes":["float32","float32"], "strides":[1,1,1,1], "padding":"SAME", "data_format":"NHWC"},
        # {"opName":"nn_conv2d", "outName":"cnn2d_nn/nhwc_b2_k3_s1_d1_SAME", "varShapes":[[2, 5, 5, 2], [3, 3, 2, 3]], "varTypes":["float32","float32"], "strides":[1,1,1,1], "padding":"SAME", "data_format":"NHWC"},
        # {"opName":"nn_conv2d", "outName":"cnn2d_nn/nhwc_b2_k2_s1_d1_SAME", "varShapes":[[2, 5, 5, 2], [2, 2, 2, 3]], "varTypes":["float32","float32"], "strides":[1,1,1,1], "padding":"SAME", "data_format":"NHWC"},
        # {"opName":"nn_conv2d", "outName":"cnn2d_nn/nhwc_b2_k2_s1_d2_SAME", "varShapes":[[2, 5, 5, 2], [2, 2, 2, 3]], "varTypes":["float32","float32"], "strides":[1,1,1,1], "padding":"SAME", "data_format":"NHWC", "dilation":[1,2,2,1]},
        # {"opName":"nn_conv2d", "outName":"cnn2d_nn/nhwc_b2_k2_s1_d1_VALID", "varShapes":[[2, 5, 5, 2], [2, 2, 2, 3]], "varTypes":["float32","float32"], "strides":[1,1,1,1], "padding":"VALID", "data_format":"NHWC"},
        # {"opName":"nn_conv2d", "outName":"cnn2d_nn/nhwc_b1_k2_s2_SAME", "varShapes":[[2, 5, 5, 2], [2, 2, 2, 3]], "varTypes":["float32","float32"], "strides":[1,2,2,1], "padding":"SAME", "data_format":"NHWC"},

        #Again, no channels_first on CPU
        # {"opName":"layers_conv2d", "outName":"cnn2d_layers/channels_last_b1_k2_s1_d1_SAME", "varShapes":[[1, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last", "dilation_rate":[1,1]},
        # {"opName":"layers_conv2d", "outName":"cnn2d_layers/channels_last_b1_k2_s1_d2_SAME", "varShapes":[[1, 5, 5, 2]], "varTypes":["float32","float32"], "filters":2, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last", "dilation_rate":[2,2]},
        # {"opName":"layers_conv2d", "outName":"cnn2d_layers/channels_last_b2_k2_s1_SAME_nobias", "varShapes":[[2, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last", "dilation_rate":[1,1], "use_bias":False},
        # {"opName":"layers_conv2d", "outName":"cnn2d_layers/channels_last_b2_k2_s1_VALID", "varShapes":[[2, 5, 5, 2]], "varTypes":["float32","float32"], "filters":2, "kernel_size":[2,2], "strides":[1,1], "padding":"VALID", "data_format":"channels_last", "dilation_rate":[1,1]},
        # {"opName":"layers_conv2d", "outName":"cnn2d_layers/channels_last_b1_k2_s2_SAME", "varShapes":[[1, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[2,2], "padding":"SAME", "data_format":"channels_last", "dilation_rate":[1,1]},
        # {"opName":"layers_conv2d", "outName":"cnn2d_layers/channels_last_b1_k2_s1_d1_SAME_sigmoid", "varShapes":[[1, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last", "dilation_rate":[1,1], "activation":tf.nn.relu},
        # {"opName":"layers_conv2d", "outName":"cnn2d_layers/channels_last_b1_k2_s1_d1_SAME_sigmoid", "varShapes":[[1, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last", "dilation_rate":[1,1], "activation":tf.nn.sigmoid},
        # {"opName":"layers_conv2d", "outName":"cnn2d_layers/channels_last_b1_k2_s1_d1_SAME_elu", "varShapes":[[1, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last", "dilation_rate":[1,1], "activation":tf.nn.elu},
        # {"opName":"layers_conv2d", "outName":"cnn2d_layers/channels_last_b1_k2_s1_d1_SAME_relu6", "varShapes":[[1, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last", "dilation_rate":[1,1], "activation":tf.nn.relu6},
        # {"opName":"layers_conv2d", "outName":"cnn2d_layers/channels_last_b1_k2_s1_d1_SAME_selu_nobias", "varShapes":[[1, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last", "dilation_rate":[1,1], "activation":tf.nn.selu, "use_bias":False},
        # {"opName":"layers_conv2d", "outName":"cnn2d_layers/channels_last_b1_k2_s1_d1_SAME_crelu", "varShapes":[[1, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last", "dilation_rate":[1,1], "activation":tf.nn.crelu},
        # {"opName":"layers_conv2d", "outName":"cnn2d_layers/channels_last_b2_k2_s1_SAME_regularizers", "varShapes":[[2, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last", "dilation_rate":[1,1],
        #     "kernel_regularizer":tf.contrib.layers.l2_regularizer(scale=0.1), "bias_regularizer":tf.contrib.layers.l1_regularizer(scale=0.2), "activity_regularizer":tf.contrib.layers.l1_l2_regularizer(scale_l1=0.1,scale_l2=0.2)},
        # {"opName":"layers_conv2d", "outName":"cnn2d_layers/channels_last_b2_k2_s1_SAME_constraints1", "varShapes":[[2, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last", "dilation_rate":[1,1],
        #     "kernel_constraint":tf.keras.constraints.MaxNorm(max_value=2), "bias_constraint":tf.keras.constraints.NonNeg()},
        # {"opName":"layers_conv2d", "outName":"cnn2d_layers/channels_last_b2_k2_s1_SAME_constraints2", "varShapes":[[2, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last", "dilation_rate":[1,1],
        #     "kernel_constraint":tf.keras.constraints.MinMaxNorm(min_value=1, max_value=2)},
        # {"opName":"layers_conv2d", "outName":"cnn2d_layers/channels_last_b2_k2_s1_SAME_constraints3", "varShapes":[[2, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last", "dilation_rate":[1,1],
        #     "kernel_constraint":tf.keras.constraints.UnitNorm()},

        # {"opName":"layers_conv2d_transpose", "outName":"conv2d_transpose/channels_last_b1_k2_s1_SAME", "varShapes":[[1, 5, 5, 2]], "varTypes":["float32","float32"], "filters":2, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last"},
        # {"opName":"layers_conv2d_transpose", "outName":"conv2d_transpose/channels_last_b2_k2_s1_SAME_nobias", "varShapes":[[2, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last", "use_bias":False},
        # {"opName":"layers_conv2d_transpose", "outName":"conv2d_transpose/channels_last_b2_k2_s1_VALID", "varShapes":[[2, 5, 5, 2]], "varTypes":["float32","float32"], "filters":2, "kernel_size":[2,2], "strides":[1,1], "padding":"VALID", "data_format":"channels_last"},
        # {"opName":"layers_conv2d_transpose", "outName":"conv2d_transpose/channels_last_b1_k2_s2_SAME", "varShapes":[[1, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[2,2], "padding":"SAME", "data_format":"channels_last"},
        # {"opName":"layers_conv2d_transpose", "outName":"conv2d_transpose/channels_last_b1_k2_s1_SAME_sigmoid", "varShapes":[[1, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last", "activation":tf.nn.relu},
        # {"opName":"layers_conv2d_transpose", "outName":"conv2d_transpose/channels_last_b1_k2_s1_SAME_sigmoid", "varShapes":[[1, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last", "activation":tf.nn.sigmoid},
        # {"opName":"layers_conv2d_transpose", "outName":"conv2d_transpose/channels_last_b1_k2_s1_SAME_elu", "varShapes":[[1, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last", "activation":tf.nn.elu},
        # {"opName":"layers_conv2d_transpose", "outName":"conv2d_transpose/channels_last_b1_k2_s1_SAME_relu6", "varShapes":[[1, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last", "activation":tf.nn.relu6},
        # {"opName":"layers_conv2d_transpose", "outName":"conv2d_transpose/channels_last_b1_k2_s1_SAME_selu_nobias", "varShapes":[[1, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last", "activation":tf.nn.selu, "use_bias":False},
        # {"opName":"layers_conv2d_transpose", "outName":"conv2d_transpose/channels_last_b1_k2_s1_SAME_crelu", "varShapes":[[1, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last", "activation":tf.nn.crelu},
        # {"opName":"layers_conv2d_transpose", "outName":"conv2d_transpose/channels_last_b2_k2_s1_SAME_regularizers", "varShapes":[[2, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2], "strides":[1,1], "padding":"SAME", "data_format":"channels_last",
        #  "kernel_regularizer":tf.contrib.layers.l2_regularizer(scale=0.1), "bias_regularizer":tf.contrib.layers.l1_regularizer(scale=0.2), "activity_regularizer":tf.contrib.layers.l1_l2_regularizer(scale_l1=0.1,scale_l2=0.2)},

        # Data format: ch_last: NDHWC, ch_first: NCDHW
        # "CPU implementation of Conv3D currently only supports the NHWC tensor format."
        #{"opName":"layers_conv3d", "outName":"cnn3d_layers/channels_first_b1_k2_s1_d1_SAME", "varShapes":[[1, 2, 5, 5, 5]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2,2], "strides":[1,1,1], "padding":"SAME", "data_format":"channels_first", "dilation_rate":[1,1,1]},
        # {"opName":"layers_conv3d", "outName":"cnn3d_layers/channels_last_b1_k2_s1_d1_SAME", "varShapes":[[1, 5, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2,2], "strides":[1,1,1], "padding":"SAME", "data_format":"channels_last", "dilation_rate":[1,1,1]},
        # {"opName":"layers_conv3d", "outName":"cnn3d_layers/channels_last_b1_k2_s1_d2_SAME", "varShapes":[[1, 5, 5, 5, 2]], "varTypes":["float32","float32"], "filters":2, "kernel_size":[2,2,2], "strides":[1,1,1], "padding":"SAME", "data_format":"channels_last", "dilation_rate":[2,2,2]},
        # {"opName":"layers_conv3d", "outName":"cnn3d_layers/channels_last_b2_k3_s1_SAME_nobias", "varShapes":[[2, 5, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[3,3,3], "strides":[1,1,1], "padding":"SAME", "data_format":"channels_last", "dilation_rate":[1,1,1], "use_bias":False},
        # {"opName":"layers_conv3d", "outName":"cnn3d_layers/channels_last_b2_k2_s1_VALID", "varShapes":[[2, 5, 5, 5, 2]], "varTypes":["float32","float32"], "filters":2, "kernel_size":[2,2,2], "strides":[1,1,1], "padding":"VALID", "data_format":"channels_last", "dilation_rate":[1,1,1]},
        # {"opName":"layers_conv3d", "outName":"cnn3d_layers/channels_last_b1_k2_s2_SAME", "varShapes":[[1, 5, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2,2], "strides":[2,2,2], "padding":"SAME", "data_format":"channels_last", "dilation_rate":[1,1,1]},
        # {"opName":"layers_conv3d", "outName":"cnn3d_layers/channels_last_b1_k2_s1_d1_SAME_sigmoid", "varShapes":[[1, 5, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2,2], "strides":[1,1,1], "padding":"SAME", "data_format":"channels_last", "dilation_rate":[1,1,1], "activation":tf.nn.relu},
        # {"opName":"layers_conv3d", "outName":"cnn3d_layers/channels_last_b2_k2_s1_SAME_regularizers", "varShapes":[[2, 5, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2,2], "strides":[1,1,1], "padding":"SAME", "data_format":"channels_last", "dilation_rate":[1,1,1],
        #     "kernel_regularizer":tf.contrib.layers.l2_regularizer(scale=0.1), "bias_regularizer":tf.contrib.layers.l1_regularizer(scale=0.2), "activity_regularizer":tf.contrib.layers.l1_l2_regularizer(scale_l1=0.1,scale_l2=0.2)},
        # {"opName":"layers_conv3d", "outName":"cnn3d_layers/channels_last_b2_k2_s1_SAME_constraints1", "varShapes":[[2, 5, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":[2,2,2], "strides":[1,1,1], "padding":"SAME", "data_format":"channels_last", "dilation_rate":[1,1,1],
        #     "kernel_constraint":tf.keras.constraints.MaxNorm(max_value=2), "bias_constraint":tf.keras.constraints.NonNeg()},

        # Max/avg pool 3d:
        # channels_first:    "Default MaxPoolingOp only supports NHWC on device type CPU"        :/
        # {"opName":"max_pooling3d", "outName":"max_pooling3d/channels_first_b1_k2_s1_SAME", "varShapes":[[1, 2, 5, 5, 5]], "varTypes":["float32"], "pooling_size":2, "stride":1, "padding":"SAME", "data_format":"channels_first"},
        # {"opName":"max_pooling3d", "outName":"max_pooling3d/channels_last_b1_k2_s1_SAME", "varShapes":[[1, 5, 5, 5, 2]], "varTypes":["float32"], "pooling_size":2, "stride":1, "padding":"SAME", "data_format":"channels_last"},
        # {"opName":"max_pooling3d", "outName":"max_pooling3d/channels_last_b2_k2_s1_SAME", "varShapes":[[2, 5, 5, 5, 2]], "varTypes":["float32"], "pooling_size":2, "stride":1, "padding":"SAME", "data_format":"channels_last"},
        # {"opName":"max_pooling3d", "outName":"max_pooling3d/channels_last_b2_k2_s1_VALID", "varShapes":[[2, 5, 5, 5, 2]], "varTypes":["float32"], "pooling_size":2, "stride":1, "padding":"VALID", "data_format":"channels_last"},
        # {"opName":"max_pooling3d", "outName":"max_pooling3d/channels_last_b1_k2_s2_SAME", "varShapes":[[1, 5, 5, 5, 2]], "varTypes":["float32"], "pooling_size":2, "stride":2, "padding":"SAME", "data_format":"channels_last"},
        #
        # #Default AvgPoolingOp only supports NHWC on device type CPU
        # {"opName":"avg_pooling3d", "outName":"avg_pooling3d/channels_first_b1_k2_s1_SAME", "varShapes":[[1, 2, 5, 5, 5]], "varTypes":["float32"], "pooling_size":2, "stride":1, "padding":"SAME", "data_format":"channels_first"},
        # {"opName":"avg_pooling3d", "outName":"avg_pooling3d/channels_last_b1_k2_s1_SAME", "varShapes":[[1, 5, 5, 5, 2]], "varTypes":["float32"], "pooling_size":2, "stride":1, "padding":"SAME", "data_format":"channels_last"},
        # {"opName":"avg_pooling3d", "outName":"avg_pooling3d/channels_last_b2_k2_s1_SAME", "varShapes":[[2, 5, 5, 5, 2]], "varTypes":["float32"], "pooling_size":2, "stride":1, "padding":"SAME", "data_format":"channels_last"},
        # {"opName":"avg_pooling3d", "outName":"avg_pooling3d/channels_last_b2_k2_s1_VALID", "varShapes":[[2, 5, 5, 5, 2]], "varTypes":["float32"], "pooling_size":2, "stride":1, "padding":"VALID", "data_format":"channels_last"},
        # {"opName":"avg_pooling3d", "outName":"avg_pooling3d/channels_last_b1_k2_s2_SAME", "varShapes":[[1, 5, 5, 5, 2]], "varTypes":["float32"], "pooling_size":2, "stride":2, "padding":"SAME", "data_format":"channels_last"},


        #Separable conv 2d - channels_last = NHWC
        # {"opName":"layers_sepconv2d", "outName":"sepconv2d_layers/channels_last_b1_k2_s1_d1_SAME_dm1", "varShapes":[[1, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1, "depth_multiplier":1},
        # {"opName":"layers_sepconv2d", "outName":"sepconv2d_layers/channels_last_b2_k2_s1_d1_SAME_dm2", "varShapes":[[2, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1, "depth_multiplier":2},
        # {"opName":"layers_sepconv2d", "outName":"sepconv2d_layers/channels_last_b2_k2_s2_d1_SAME_dm2", "varShapes":[[2, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":2, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1, "depth_multiplier":2},
        # {"opName":"layers_sepconv2d", "outName":"sepconv2d_layers/channels_last_b2_k2_s1_d2_SAME_dm2", "varShapes":[[2, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":2, "depth_multiplier":2},
        # {"opName":"layers_sepconv2d", "outName":"sepconv2d_layers/channels_last_b2_k2_s1_d1_SAME_dm1_sigm", "varShapes":[[2, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1, "depth_multiplier":1, "activation":tf.nn.tanh},
        # {"opName":"layers_sepconv2d", "outName":"sepconv2d_layers/channels_last_b2_k2_s1_d1_SAME_dm2_sigm_nobias", "varShapes":[[2, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"SAME", "data_format":"channels_last", "dilation_rate":1, "depth_multiplier":2, "activation":tf.nn.tanh, "use_bias":False},
        # {"opName":"layers_sepconv2d", "outName":"sepconv2d_layers/channels_last_b1_k2_s1_d1_VALID_dm1", "varShapes":[[1, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"VALID", "data_format":"channels_last", "dilation_rate":1, "depth_multiplier":1},
        # {"opName":"layers_sepconv2d", "outName":"sepconv2d_layers/channels_last_b2_k2_s1_d1_VALID_dm2", "varShapes":[[2, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"VALID", "data_format":"channels_last", "dilation_rate":1, "depth_multiplier":2},
        # {"opName":"layers_sepconv2d", "outName":"sepconv2d_layers/channels_last_b2_k2_s2_d1_VALID_dm2", "varShapes":[[2, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":2, "padding":"VALID", "data_format":"channels_last", "dilation_rate":1, "depth_multiplier":2},
        # {"opName":"layers_sepconv2d", "outName":"sepconv2d_layers/channels_last_b2_k2_s1_d2_VALID_dm2", "varShapes":[[2, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"VALID", "data_format":"channels_last", "dilation_rate":2, "depth_multiplier":2},
        # {"opName":"layers_sepconv2d", "outName":"sepconv2d_layers/channels_last_b2_k2_s1_d1_VALID_dm1_sigm", "varShapes":[[2, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"VALID", "data_format":"channels_last", "dilation_rate":1, "depth_multiplier":1, "activation":tf.nn.tanh},
        # {"opName":"layers_sepconv2d", "outName":"sepconv2d_layers/channels_last_b2_k2_s1_d1_VALID_dm2_sigm_nobias", "varShapes":[[2, 5, 5, 2]], "varTypes":["float32","float32"], "filters":3, "kernel_size":2, "strides":1, "padding":"VALID", "data_format":"channels_last", "dilation_rate":1, "depth_multiplier":2, "activation":tf.nn.tanh, "use_bias":False},


        #Batch norm - 2d
        # {"opName":"batchnorm", "outName":"batchnorm/rank2_batch2_sz5_fused", "varShapes":[[2, 5]], "varTypes":["float32"], "axis":1, "fused":True},
        # {"opName":"batchnorm", "outName":"batchnorm/rank2_batch4_sz5_noFused", "varShapes":[[4, 5]], "varTypes":["float32"], "axis":1, "fused":False},
        # {"opName":"batchnorm", "outName":"batchnorm/rank2_batch3_sz5_fused_m50_e01", "varShapes":[[3, 5]], "varTypes":["float32"], "axis":1, "fused":True, "momentum":0.5, "epsilon":0.1},
        # {"opName":"batchnorm", "outName":"batchnorm/rank2_batch4_sz5_noFused_m50e01", "varShapes":[[4, 5]], "varTypes":["float32"], "axis":1, "fused":False, "momentum":0.5, "epsilon":0.1},
        #
        # #Batch norm - 3d input (time series) - NCW = [mb, size, length]
        # {"opName":"batchnorm", "outName":"batchnorm/rank3_ncw_batch2_sz5_fused", "varShapes":[[2, 5, 5]], "varTypes":["float32"], "axis":1, "fused":True},
        # {"opName":"batchnorm", "outName":"batchnorm/rank3_ncw_batch4_sz5_noFused", "varShapes":[[4, 5, 5]], "varTypes":["float32"], "axis":1, "fused":False},
        # {"opName":"batchnorm", "outName":"batchnorm/rank3_ncw_batch3_sz5_fused_m50_e01", "varShapes":[[3, 5, 3]], "varTypes":["float32"], "axis":1, "fused":True, "momentum":0.5, "epsilon":0.1},
        # {"opName":"batchnorm", "outName":"batchnorm/rank3_ncw_batch4_sz5_noFused_m50e01", "varShapes":[[4, 5, 1]], "varTypes":["float32"], "axis":1, "fused":False, "momentum":0.5, "epsilon":0.1},
        # {"opName":"batchnorm", "outName":"batchnorm/rank3_nwc_batch2_sz5_fused", "varShapes":[[2, 5, 5]], "varTypes":["float32"], "axis":2, "fused":True},
        # {"opName":"batchnorm", "outName":"batchnorm/rank3_nwc_batch4_sz5_noFused", "varShapes":[[4, 5, 5]], "varTypes":["float32"], "axis":2, "fused":False},
        # {"opName":"batchnorm", "outName":"batchnorm/rank3_nwc_batch3_sz5_fused_m50_e01", "varShapes":[[3, 5, 3]], "varTypes":["float32"], "axis":2, "fused":True, "momentum":0.5, "epsilon":0.1},
        # {"opName":"batchnorm", "outName":"batchnorm/rank3_nwc_batch4_sz5_noFused_m50e01", "varShapes":[[4, 5, 1]], "varTypes":["float32"], "axis":2, "fused":False, "momentum":0.5, "epsilon":0.1},

        #Batch norm - 4d input (2d CNN)
        #Can't do fused + nchw(axis=1)
        # # {"opName":"batchnorm", "outName":"batchnorm/rank4_nchw_batch2_sz5_fused", "varShapes":[[2, 5, 5, 3]], "varTypes":["float32"], "axis":1, "fused":True},
        # {"opName":"batchnorm", "outName":"batchnorm/rank4_nchw_batch4_sz5_noFused", "varShapes":[[4, 5, 5, 3]], "varTypes":["float32"], "axis":1, "fused":False},
        # # {"opName":"batchnorm", "outName":"batchnorm/rank4_nchw_batch3_sz5_fused_m50_e01", "varShapes":[[3, 5, 3, 5]], "varTypes":["float32"], "axis":1, "fused":True, "momentum":0.5, "epsilon":0.1},
        # {"opName":"batchnorm", "outName":"batchnorm/rank4_nchw_batch4_sz5_noFused_m50e01", "varShapes":[[4, 5, 5, 1]], "varTypes":["float32"], "axis":1, "fused":False, "momentum":0.5, "epsilon":0.1},
        # {"opName":"batchnorm", "outName":"batchnorm/rank4_nhwc_batch2_sz5_fused", "varShapes":[[2, 5, 5, 5]], "varTypes":["float32"], "axis":3, "fused":True},
        # {"opName":"batchnorm", "outName":"batchnorm/rank4_nhwc_batch4_sz5_noFused", "varShapes":[[4, 5, 5, 5]], "varTypes":["float32"], "axis":3, "fused":False},
        # {"opName":"batchnorm", "outName":"batchnorm/rank4_nhwc_batch3_sz5_fused_m50_e01", "varShapes":[[3, 5, 5, 3]], "varTypes":["float32"], "axis":3, "fused":True, "momentum":0.5, "epsilon":0.1},
        # {"opName":"batchnorm", "outName":"batchnorm/rank4_nhwc_batch4_sz5_noFused_m50e01", "varShapes":[[4, 5, 5, 1]], "varTypes":["float32"], "axis":3, "fused":False, "momentum":0.5, "epsilon":0.1},

        #Leaky RELU
        # {"opName":"leaky_relu", "outName":"leaky_relu/rank2_a0", "varShapes":[[4, 5]], "varTypes":["float32"], "varInit":["stdnormal"], "alpha":0.0},
        # {"opName":"leaky_relu", "outName":"leaky_relu/rank4_a05", "varShapes":[[4, 5]], "varTypes":["float32"], "varInit":["stdnormal"], "alpha":0.5},
        # {"opName":"leaky_relu", "outName":"leaky_relu/rank4_a0", "varShapes":[[4, 5, 5, 1]], "varTypes":["float32"], "varInit":["stdnormal"], "alpha":0.0},
        # {"opName":"leaky_relu", "outName":"leaky_relu/rank4_a02", "varShapes":[[4, 5, 5, 1]], "varTypes":["float32"], "varInit":["stdnormal"], "alpha":0.2},

        #Embedding lookup
        # {"opName":"embedding_lookup", "outName":"embedding_lookup/rank2_single_div_nomaxnorm", "varShapes":[[10, 5],[4]], "varTypes":["float32","int32"], "varInit":["uniform","uniform_int10"], "partition_strategy":"div", "max_norm":None},
        # {"opName":"embedding_lookup", "outName":"embedding_lookup/rank2_single_mod_maxnorm1", "varShapes":[[10, 5],[4]], "varTypes":["float32","int32"], "varInit":["uniform","uniform_int10"], "partition_strategy":"mod", "max_norm":1.0},
        # {"opName":"embedding_lookup", "outName":"embedding_lookup/rank2_multiple_div_nomaxnorm", "varShapes":[[4, 5],[3,5],[3,5],[4]], "varTypes":["float32","float32","float32","int32"], "varInit":["uniform","uniform","uniform","uniform_int10"], "partition_strategy":"div", "max_norm":None},
        # {"opName":"embedding_lookup", "outName":"embedding_lookup/rank2_multiple_mod_maxnorm1", "varShapes":[[4, 5],[3,5],[3,5],[4]], "varTypes":["float32","float32","float32","int32"], "varInit":["uniform","uniform","uniform","uniform_int10"], "partition_strategy":"mod", "max_norm":1.0},
        # {"opName":"embedding_lookup", "outName":"embedding_lookup/rank4_single_div_nomaxnorm", "varShapes":[[10, 5],[4]], "varTypes":["float32","int32"], "varInit":["uniform","uniform_int10"], "partition_strategy":"div", "max_norm":None},
        # {"opName":"embedding_lookup", "outName":"embedding_lookup/rank4_single_mod_maxnorm1", "varShapes":[[10, 2, 3, 4],[4]], "varTypes":["float32","int32"], "varInit":["uniform","uniform_int10"], "partition_strategy":"mod", "max_norm":1.0},
        # {"opName":"embedding_lookup", "outName":"embedding_lookup/rank4_multiple_div_nomaxnorm", "varShapes":[[4,2,3,4],[3,2,3,4],[3,2,3,4],[4]], "varTypes":["float32","float32","float32","int32"], "varInit":["uniform","uniform","uniform","uniform_int10"], "partition_strategy":"div", "max_norm":None},
        # {"opName":"embedding_lookup", "outName":"embedding_lookup/rank4_multiple_mod_maxnorm1", "varShapes":[[4,2,3,4],[3,2,3,4],[3,2,3,4],[4]], "varTypes":["float32","float32","float32","int32"], "varInit":["uniform","uniform","uniform","uniform_int10"], "partition_strategy":"mod", "max_norm":1.0},

        # {"opName":"l2_normalize", "outName":"l2_normalize/rank2_e0", "varShapes":[[4, 5]], "varTypes":["float32"], "varInit":["uniform"], "axis":1, "epsilon":0.0},
        # {"opName":"l2_normalize", "outName":"l2_normalize/rank2_e05", "varShapes":[[4, 5]], "varTypes":["float32"], "varInit":["uniform"], "axis":1, "epsilon":0.5},
        # {"opName":"l2_normalize", "outName":"l2_normalize/rank4_e0_d1", "varShapes":[[3,2,3,4]], "varTypes":["float32"], "varInit":["uniform"], "axis":1, "epsilon":0.0},
        # {"opName":"l2_normalize", "outName":"l2_normalize/rank4_e05_d123", "varShapes":[[3,3,4,5]], "varTypes":["float32"], "varInit":["uniform"], "axis":[1,2,3], "epsilon":0.5}

        # {"opName":"lrn", "outName":"lrn/dr5_b1_a1_b05", "varShapes":[[2, 4, 4, 8]], "varTypes":["float32"], "varInit":["uniform"], "depth_radius":5, "bias":1.0, "alpha":1.0, "beta":0.5},
        # {"opName":"lrn", "outName":"lrn/dr3_b05_a05_b02", "varShapes":[[2, 4, 4, 8]], "varTypes":["float32"], "varInit":["uniform"], "depth_radius":3, "bias":0.5, "alpha":0.5, "beta":0.2},

        #Dropouts. Note that due to random nature - we need to validate these differently than simply "samediff output equals tensorflow output"
        # {"opName":"layers_dropout", "outName":"layers_dropout/rank2_d05_train", "varShapes":[[4,5]], "varTypes":["float32"], "varInit":["uniform"], "rate":0.5, "training":True},
        # {"opName":"layers_dropout", "outName":"layers_dropout/rank2_d05_test", "varShapes":[[4,5]], "varTypes":["float32"], "varInit":["uniform"], "rate":0.5, "training":False},
        # {"opName":"layers_dropout", "outName":"layers_dropout/rank2_d01_train", "varShapes":[[4,5]], "varTypes":["float32"], "varInit":["uniform"], "rate":0.1, "training":True},
        # {"opName":"layers_dropout", "outName":"layers_dropout/rank2_d01_test", "varShapes":[[4,5]], "varTypes":["float32"], "varInit":["uniform"], "rate":0.1, "training":False},
        # {"opName":"layers_dropout", "outName":"layers_dropout/rank2_d09_train", "varShapes":[[4,5]], "varTypes":["float32"], "varInit":["uniform"], "rate":0.9, "training":True},
        # {"opName":"layers_dropout", "outName":"layers_dropout/rank3_d05_train_mask1", "varShapes":[[4,5,6]], "varTypes":["float32"], "varInit":["uniform"], "rate":0.5, "training":True, "noise_shape":[4,1,6]},
        # {"opName":"layers_dropout", "outName":"layers_dropout/rank3_d05_train_mask2", "varShapes":[[4,5,6]], "varTypes":["float32"], "varInit":["uniform"], "rate":0.5, "training":True, "noise_shape":[4,5,1]},
        # {"opName":"layers_dropout", "outName":"layers_dropout/rank3_d05_test", "varShapes":[[4,5,6]], "varTypes":["float32"], "varInit":["uniform"], "rate":0.5, "training":False},
        # {"opName":"layers_dropout", "outName":"layers_dropout/rank4_d05_train", "varShapes":[[2,5,5,3]], "varTypes":["float32"], "varInit":["uniform"], "rate":0.5, "training":True},
        # {"opName":"layers_dropout", "outName":"layers_dropout/rank4_d05_train_mask", "varShapes":[[2,5,5,3]], "varTypes":["float32"], "varInit":["uniform"], "rate":0.5, "training":True, "noise_shape":[2,1,1,3]},

        # {"opName":"contrib_nn_alpha_dropout", "outName":"alpha_dropout/rank2_p05", "varShapes":[[4,5]], "varTypes":["float32"], "varInit":["uniform"], "keep_prob":0.5},
        # {"opName":"contrib_nn_alpha_dropout", "outName":"alpha_dropout/rank2_p01", "varShapes":[[4,5]], "varTypes":["float32"], "varInit":["uniform"], "keep_prob":0.1},
        # {"opName":"contrib_nn_alpha_dropout", "outName":"alpha_dropout/rank2_p09", "varShapes":[[4,5]], "varTypes":["float32"], "varInit":["uniform"], "keep_prob":0.9},
        # {"opName":"contrib_nn_alpha_dropout", "outName":"alpha_dropout/rank3_p05_mask1", "varShapes":[[4,5,6]], "varTypes":["float32"], "varInit":["uniform"], "keep_prob":0.5, "noise_shape":[4,1,6]},
        # {"opName":"contrib_nn_alpha_dropout", "outName":"alpha_dropout/rank3_p05_mask2", "varShapes":[[4,5,6]], "varTypes":["float32"], "varInit":["uniform"], "keep_prob":0.5, "noise_shape":[4,5,1]},
        # {"opName":"contrib_nn_alpha_dropout", "outName":"alpha_dropout/rank4_p05", "varShapes":[[2,5,5,3]], "varTypes":["float32"], "varInit":["uniform"], "keep_prob":0.5},
        # {"opName":"contrib_nn_alpha_dropout", "outName":"alpha_dropout/rank4_p05_mask", "varShapes":[[2,5,5,3]], "varTypes":["float32"], "varInit":["uniform"], "keep_prob":0.5, "noise_shape":[2,1,1,3]},

        #Meshgrid - seems like TF doesn't like things like [[3], [4]] - "ValueError: Dimension 0 in both shapes must be equal, but are 3 and 4. Shapes are [3] and [4]."
        # {"opName":"meshgrid", "outName":"meshgrid/n1_xy", "varShapes":[[3]], "varTypes":["float32"], "varInit":["uniform"], "indexing":"xy"},
        # {"opName":"meshgrid", "outName":"meshgrid/n1_ij", "varShapes":[[3]], "varTypes":["float32"], "varInit":["uniform"], "indexing":"ij"},
        # {"opName":"meshgrid", "outName":"meshgrid/n2_xy", "varShapes":[[3],[3]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"], "indexing":"xy"},
        # {"opName":"meshgrid", "outName":"meshgrid/n2_ij", "varShapes":[[3],[3]], "varTypes":["float32","float32"], "varInit":["uniform","uniform"], "indexing":"ij"},
        # {"opName":"meshgrid", "outName":"meshgrid/n3_xy", "varShapes":[[3],[3],[3]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"], "indexing":"xy"},
        # {"opName":"meshgrid", "outName":"meshgrid/n3_ij", "varShapes":[[3],[3],[3]], "varTypes":["float32","float32","float32"], "varInit":["uniform","uniform","uniform"], "indexing":"ij"},
        # {"opName":"meshgrid", "outName":"meshgrid/n4_xy", "varShapes":[[3],[3],[3],[3]], "varTypes":["float32","float32","float32","float32"], "varInit":["uniform","uniform","uniform","uniform"],"indexing":"xy"},
        # {"opName":"meshgrid", "outName":"meshgrid/n4_ij", "varShapes":[[3],[3],[3],[3]], "varTypes":["float32","float32","float32","float32"], "varInit":["uniform","uniform","uniform","uniform"],"indexing":"ij"}

        # {"opName":"eye", "outName":"eye/e22", "varShapes":[], "varTypes":[], "varInit":[], "num_rows":2, "num_columns":2},
        # {"opName":"eye", "outName":"eye/e23", "varShapes":[], "varTypes":[], "varInit":[], "num_rows":2, "num_columns":3},
        # {"opName":"eye", "outName":"eye/e32", "varShapes":[], "varTypes":[], "varInit":[], "num_rows":3, "num_columns":2},
        {"opName":"eye", "outName":"eye/e22_b1", "varShapes":[[1]], "varTypes":["int32"], "varInit":["one"], "num_rows":2, "num_columns":2},
        {"opName":"eye", "outName":"eye/e23_b2", "varShapes":[[1]], "varTypes":["int32"], "varInit":["two"], "num_rows":2, "num_columns":3},
        {"opName":"eye", "outName":"eye/e32_b22", "varShapes":[[2]], "varTypes":["int32"], "varInit":["two"], "num_rows":3, "num_columns":2},

           ]



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
