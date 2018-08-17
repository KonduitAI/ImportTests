import tensorflow as tf
import numpy as np


class OpCreator:
    def __init__(self, op):
        self.op = op
        self.node_num = 0

    def setVars(self, vars):
        self.vars = vars

    def setPlaceholders(self, placeholders):
        self.placeholders = placeholders

    def execute(self, some_op):
        self.node_num += 1
        method_name = 'execute_' + some_op
        try:
            method = getattr(self, method_name)
        except AttributeError:
            print(method_name, "not found")
        else:
            return method()

    def execute_reduce_sum(self):
        return [tf.reduce_sum(self.vars[0], axis=self.axis, keepdims=self.extra.get("keepdims", False), name="reduce_sum" + str(self.node_num))]

    def execute_segment_max(self):
        return [tf.segment_max(data=self.vars[0], segment_ids=self.vars[1])]

    def execute_segment_min(self):
        return [tf.segment_min(data=self.vars[0], segment_ids=self.vars[1])]

    def execute_segment_mean(self):
        return [tf.segment_mean(data=self.vars[0], segment_ids=self.vars[1])]

    def execute_segment_prod(self):
        return [tf.segment_prod(data=self.vars[0], segment_ids=self.vars[1])]

    def execute_segment_sum(self):
        return [tf.segment_sum(data=self.vars[0], segment_ids=self.vars[1])]

    def execute_space_to_batch(self):
        return [tf.space_to_batch(input=self.vars[0], paddings=self.vars[1], block_size=2)]

    def execute_space_to_depth(self):
        return [tf.space_to_depth(input=self.vars[0], block_size=2, data_format=self.op["data_format"])]

    def execute_batch_to_space(self):
        return [tf.batch_to_space(input=self.vars[0], crops=self.vars[1], block_size=2)]

    def execute_depth_to_space(self):
        return [tf.depth_to_space(input=self.vars[0], block_size=2, data_format=self.op["data_format"])]

    def execute_size(self):
        temp = tf.add(self.vars[0], 1.0)
        return [tf.add(tf.size(input=temp), 1)]

    def execute_shape(self):
        temp = tf.add(self.vars[0], 1.0)
        return [tf.add(tf.shape(input=temp), 1)]

    def execute_shapen(self):
        out = tf.shape_n(input=self.vars)
        #Concat multiple outputs to avoid graph saving issue
        return [tf.concat(out, axis=0)]

    def execute_matrix_inverse(self):
        return [tf.matrix_inverse(input=self.vars[0])]

    def execute_pad(self):
        if(len(self.vars) > 2):
            return [tf.pad(tensor=self.vars[0], paddings=self.vars[1], constant_values=self.vars[2], mode = self.op["mode"])]
        else:
            return [tf.pad(tensor=self.vars[0], paddings=self.vars[1], mode=self.op["mode"])]

    def execute_unique(self):
        #Hack for multi-output saving issue: concat
        temp = tf.unique(self.vars[0])
        toConcat = []
        toConcat.append(temp[0])
        toConcat.append(tf.cast(temp[1], dtype=tf.float32))
        return [tf.concat(toConcat, axis=0)]

    def execute_unique_with_counts(self):
        temp = tf.unique_with_counts(self.vars[0])
        toConcat = []
        toConcat.append(temp[0])
        toConcat.append(tf.cast(temp[1], dtype=tf.float32))
        toConcat.append(tf.cast(temp[2], dtype=tf.float32))
        return [tf.concat(toConcat,axis=0)]

    def execute_topk(self):
        temp = tf.nn.top_k(input=self.vars[0], k=self.op["k"], sorted=self.op["sorted"])
        toConcat = []
        toConcat.append(temp[0])
        toConcat.append(tf.cast(temp[1], dtype=tf.float32))
        #Concat multiple outputs to avoid graph saving issue. Note that values and indices have same shape
        return [tf.concat(toConcat, axis=0)]

    def execute_in_top_k(self):
        return [tf.nn.in_top_k(predictions=self.vars[0], targets=self.vars[1], k=self.op["k"])]

    def execute_matrix_determinant(self):
        return [tf.matrix_determinant(input=self.vars[0])]

    def execute_matrix_set_diag(self):
        return [tf.matrix_set_diag(input=self.vars[0], diagonal=self.vars[1])]

    def execute_identity_n(self):
        return tf.identity_n(self.vars)

    def execute_zeta(self):
        x = tf.add(self.vars[0], 1.0)    #x values must be > 1
        return [tf.zeta(x=x, q=self.vars[1])]

    def execute_confusion_matrix(self):
        weights = None
        if(len(self.vars) > 2):
            weights = self.vars[2]
        return [tf.confusion_matrix(labels=self.vars[0], predictions=self.vars[1], num_classes=self.op["num_classes"], weights=weights)]

    def execute_stack(self):
        return [tf.stack(values=self.vars, axis=self.op["axis"])]

    def execute_parallel_stack(self):
        return [tf.parallel_stack(values=self.vars)]

    def execute_accumulate_n(self):
        return [tf.accumulate_n(self.vars)]

    def execute_angle(self):
        return [tf.add(tf.angle(self.vars[0]), 1.0)]

    def execute_approximate_equal(self):
        return [tf.approximate_equal(self.vars[0], self.vars[1])]

    def execute_matmul(self):
        ta = self.op.get("transpose_a", False)
        tb = self.op.get("transpose_b", False)
        print(self.op)
        print("ta = ",ta)
        print("tb = ",tb)
        return [tf.matmul(self.vars[0], self.vars[1], transpose_a=ta, transpose_b=tb, name = "matmul-" + str(self.node_num))]

    def execute_matrix_diag_part(self):
        return [tf.matrix_diag_part(self.vars[0])]

    def execute_svd(self):
        shapes = self.op["varShapes"]
        if(shapes[len(shapes)-1] != shapes[len(shapes)-2]):
            raise ValueError("Only square inputs currently supported due to multiple outputs issue")

        svd = tf.svd(tensor=self.vars[0], full_matrices=self.op["full_matrices"], compute_uv=self.op["compute_uv"])
        #Outputs: If compute_uv is false, only one output
        if(self.op["compute_uv"] is False or len(svd) == 1):
            if(isinstance(svd, list)):
                return svd
            return [svd]

        #Multiple outputs issue: s, shape [..., P], u shape [..., M, P] or [..., M, M]
        # v shape [..., N,P] or [..., N, N]
        # Where P is min(M,N)
        #Workaround for multiple outputs saving issue: if m=n, can add u and v... then need to reshape s to [..., M, 1] and broadcast add...
        s = svd[0]
        u = svd[1]
        v = svd[2]
        s = tf.expand_dims(s, -1)
        return [s + u + v]
