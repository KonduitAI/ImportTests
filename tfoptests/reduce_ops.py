import tensorflow as tf


class ReduceOps:
    def __init__(self, vars, shapes, types, axis, extra):
        self.vars = vars
        self.shapes = shapes
        self.types = types
        self.axis = axis
        self.node_num = 0
        self.extra = extra

    def set_a(self, a):
        self.a = a

    def set_b(self, b):
        self.b = b

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

    def execute_reduce_max(self):
        return [tf.reduce_max(self.vars[0], axis=self.axis, keepdims=self.extra.get("keepdims", False), name="reduce_max" + str(self.node_num))]

    def execute_reduce_min(self):
        return [tf.reduce_min(self.vars[0], axis=self.axis, keepdims=self.extra.get("keepdims", False), name="reduce_min" + str(self.node_num))]

    def execute_reduce_mean(self):
        return [tf.reduce_mean(self.vars[0], axis=self.axis, keepdims=self.extra.get("keepdims", False), name="reduce_mean" + str(self.node_num))]

    def execute_reduce_prod(self):
        return [tf.reduce_prod(self.vars[0], axis=self.axis, keepdims=self.extra.get("keepdims", False), name="reduce_prod" + str(self.node_num))]

    def execute_is_non_decreasing(self):
        return [tf.is_non_decreasing(self.vars[0], name="is_non_decreasing-" + str(self.node_num))]

    def execute_argmax(self):
        return [tf.argmax(self.vars[0], axis=self.axis, name="argmax-" + str(self.node_num))]

    def execute_argmin(self):
        return [tf.argmin(self.vars[0], axis=self.axis, name="argmin-" + str(self.node_num))]

    def execute_add_n(self):
        return [tf.add_n(self.vars, name="add_n-" + str(self.node_num))]

    def execute_moments(self):
        return tf.nn.moments(self.vars[0], axes=self.axis, keep_dims=self.extra.get("keepdims", False))

    def execute_count_nonzero(self):
        return [tf.count_nonzero(self.vars[0], axis=self.axis, name="count_nonzero-" + str(self.node_num))]

    def execute_normalize_moments(self):
        shift = None
        if(len(self.vars) > 3):
            shift = self.vars[3]
        return tf.nn.normalize_moments(counts=self.vars[0], mean_ss=self.vars[1], variance_ss=self.vars[2], shift=shift)

    def execute_scatter_add(self):
        #r = tf.add(self.vars[0], 1.0)
        #return [tf.scatter_add(ref=self.vars[0], indices=self.vars[1], updates=self.vars[2], name="scatter_add-" + str(self.node_num))]
        return [tf.scatter_add(ref=self.vars[0], indices=tf.zeros([], dtype=tf.int32), updates=tf.random_normal([3]), name="scatter_add-" + str(self.node_num))]


