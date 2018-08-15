import tensorflow as tf
import numpy as np


class VarInitializer:
    #def __init__(self):
    #    self.node_num = 0


    def newVar(self, initType, shape, dtype, name):
        method_name = "var_" + initType
        try:
            method = getattr(self, method_name)
        except AttributeError:
            print(method_name, "not found")
        else:
            return method(shape, dtype, name)

    def var_zero(self, shape, dtype, n):
        return tf.Variable(tf.zeros(shape=shape, dtype=dtype), name=n)

    def var_one(self, shape, dtype, n):
        return tf.Variable(tf.ones(shape=shape, dtype=dtype), name=n)

    def var_range(self, shape, dtype, n):
        return tf.Variable(tf.reshape(tf.range(start=0, limit=np.prod(shape), delta=1, dtype=dtype), shape), name=n)

    def var_uniform(self, shape, dtype, n):
        return tf.Variable(tf.random_uniform(shape), dtype, name=n)

    def var_uniform10(self, shape, dtype, n):
        return tf.Variable(tf.random_uniform(shape, minval=0, maxval=10), dtype, name=n)

    def var_uniform_int10(self, shape, dtype, n):
        return tf.Variable(tf.floor(tf.random_uniform(shape, minval=0, maxval=10)), dtype, name=n)

    def var_segment3(self, shape, dtype, n):
        return self.var_segmentN(3, shape, dtype, n)

    def var_segment5(self, shape, dtype, n):
        return self.var_segmentN(5, shape, dtype, n)

    def var_segmentN(self, numSegments, shape, dtype, n):
        length = np.prod(shape)
        numPerSegment = length // numSegments
        segmentIds = []
        for i in range(length):
            segmentIds.append(min(numSegments-1, i//numPerSegment))
        return tf.Variable(tf.constant(value=segmentIds, dtype=dtype, shape=shape), name=n)
