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

