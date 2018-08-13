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
        return tf.reduce_sum(self.vars[0], axis=self.axis, keepdims=extra.get("keepdims", False), reduction_indices=self.axis, name="reduce_sum" + str(self.node_num))
