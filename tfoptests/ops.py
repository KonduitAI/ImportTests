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

