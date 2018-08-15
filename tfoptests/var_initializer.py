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

    def var_range(self, shape, dtype, n):
        return tf.Variable(tf.reshape(tf.range(start=0, limit=np.prod(shape), delta=1, dtype=dtype), shape), name=n)

    def var_uniform(self, shape, dtype, n):
        return tf.Variable(tf.random_uniform(shape), dtype, name=n)

    def var_segment5(self, shape, dtype, n):
        #TODO
        return tf.Variable(tf.reshape(tf.range(start=0, limit=np.prod(shape), delta=1, dtype=dtype), shape), name=n)
