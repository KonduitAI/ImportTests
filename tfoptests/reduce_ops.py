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
