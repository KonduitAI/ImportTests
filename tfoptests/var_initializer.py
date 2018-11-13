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

    def var_zeros(self, shape, dtype, n):
        return self.var_zero(shape, dtype, n)

    def var_zero(self, shape, dtype, n):
        return tf.Variable(tf.zeros(shape=shape, dtype=dtype), name=n)

    def var_one(self, shape, dtype, n):
        return tf.Variable(tf.ones(shape=shape, dtype=dtype), name=n)

    def var_two(self, shape, dtype, n):
        return tf.Variable(tf.cast(tf.fill(dims=shape, value=2), dtype=dtype), name=n)

    def var_three(self, shape, dtype, n):
        return tf.Variable(tf.cast(tf.fill(dims=shape, value=3), dtype=dtype), name=n)

    def var_four(self, shape, dtype, n):
        return tf.Variable(tf.cast(tf.fill(dims=shape, value=4), dtype=dtype), name=n)

    def var_ten(self, shape, dtype, n):
        return tf.Variable(tf.ones(shape=shape, dtype=dtype) * 10, name=n)

    def var_range(self, shape, dtype, n):
        return tf.Variable(tf.reshape(tf.range(start=0, limit=np.prod(shape), delta=1, dtype=dtype), shape), name=n)

    def var_stdnormal(self, shape, dtype, n):
        return tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=1.0, dtype=dtype), dtype=dtype, name=n)

    def var_uniform(self, shape, dtype, n):
        return tf.Variable(tf.random_uniform(shape), dtype, name=n)

    def var_uniform_m1_1(self, shape, dtype, n):
        return tf.Variable(tf.random_uniform(shape, minval=-1, maxval=1), dtype, name=n)

    def var_uniform10(self, shape, dtype, n):
        return tf.Variable(tf.random_uniform(shape, minval=0, maxval=10, dtype=dtype), dtype, name=n)

    def var_uniform_int3(self, shape, dtype, n):
        if(dtype == tf.int32):
            return tf.Variable(tf.random_uniform(shape, minval=0, maxval=3, dtype=dtype), dtype, name=n)
        else:
            return tf.Variable(tf.floor(tf.random_uniform(shape, minval=0, maxval=3, dtype=dtype)), dtype, name=n)

    def var_uniform_int5(self, shape, dtype, n):
        if(dtype == tf.int32):
            return tf.Variable(tf.random_uniform(shape, minval=0, maxval=5, dtype=dtype), dtype, name=n)
        else:
            return tf.Variable(tf.floor(tf.random_uniform(shape, minval=0, maxval=5, dtype=dtype)), dtype, name=n)

    def var_uniform_int10(self, shape, dtype, n):
        if(dtype == tf.int32):
            return tf.Variable(tf.random_uniform(shape, minval=0, maxval=10, dtype=dtype), dtype, name=n)
        else:
            return tf.Variable(tf.floor(tf.random_uniform(shape, minval=0, maxval=10, dtype=dtype)), dtype, name=n)

    def var_uniform_sparse(self, shape, dtype, n):
        values = tf.random_uniform(shape) * tf.cast((tf.random_uniform(shape) < 0.5), dtype=tf.float32)
        return tf.Variable(values, dtype, name=n)

    def var_fixed_m1_1(self, shape, dtype, n):
        if(len(shape) is not 1 or shape[0] is not 2):
            raise ValueError("Shape must be exactly [2]")
        return tf.Variable([-1, 1], dtype=dtype, name=n)

    def var_fixed_2_1(self, shape, dtype, n):
        if(len(shape) is not 1 or shape[0] is not 2):
            raise ValueError("Shape must be exactly [2]")
        return tf.Variable([2,1], dtype=dtype, name=n)

    def var_fixed_5_3(self, shape, dtype, n):
        if(len(shape) is not 1 or shape[0] is not 2):
            raise ValueError("Shape must be exactly [2]")
        return tf.Variable([5, 3], dtype=dtype, name=n)

    def var_fixed_2_2_4(self, shape, dtype, n):
        if(len(shape) is not 1 or shape[0] is not 3):
            raise ValueError("Shape must be exactly [3]")
        return tf.Variable([2,2,4], dtype=dtype, name=n)

    def var_fixed_2_1_4(self, shape, dtype, n):
        if(len(shape) is not 1 or shape[0] is not 3):
            raise ValueError("Shape must be exactly [3]")
        return tf.Variable([2,1,4], dtype=dtype, name=n)

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

    def var_bernoulli(self, shape, dtype, n):
        #Random 0 or 1
        return tf.cast((tf.random_uniform(shape) < 0.5), dtype=dtype)

    def var_onehot(self, shape, dtype, n):
        if(len(shape) is not 2):
            raise ValueError("Only rank 2 input supported so far")

        # x = np.zeros(shape)
        # y = np.random.choice(shape[0], shape[1])
        # x[np.arange(shape[0]), y] = 1

        x = np.eye(shape[1])[np.random.choice(shape[1], size=shape[0])]

        return tf.Variable(x, dtype=dtype, name=n)


    def var_boolean(self, shape, dtype, n):
        print(dtype)
        # if(dtype is not tf.bool):
        return tf.Variable(tf.random_uniform(shape) >= 0.5, dtype=dtype)

    def var_booleanFalse(self, shape, dtype, n):
        return tf.Variable(tf.cast(tf.fill(dims=shape, value=False), dtype=dtype), name=n)

    def var_booleanTrue(self, shape, dtype, n):
        return tf.Variable(tf.cast(tf.fill(dims=shape, value=True), dtype=dtype), name=n)


    def newPlaceholder(self, initType, shape, dtype, name):
        method_name = "placeholder_" + initType
        try:
            method = getattr(self, method_name)
        except AttributeError:
            print(method_name, "not found")
        else:
            return method(shape, dtype, name)

    def placeholder_zero(self, shape, dtype, n):
        return [tf.placeholder(dtype=dtype, shape=shape, name=n),
                np.zeros(shape, dtype.as_numpy_dtype())]

    def placeholder_one(self, shape, dtype, n):
        return [tf.placeholder(dtype=dtype, shape=shape, name=n),
                np.ones(shape, dtype.as_numpy_dtype())]

