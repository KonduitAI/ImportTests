import tensorflow as tf


class DifferentiableMathOps:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.node_num = 0

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
            print method_name, "not found"
        else:
            return method()

    def execute_add(self):
        return tf.add(self.a, self.b, name="add" + str(self.node_num))

    def execute_add_n(self):
        return tf.add_n([self.a, self.b], name="add_n" + str(self.node_num))

    def execute_max(self):
        return tf.maximum(self.a, self.b, name="maximum" + str(self.node_num))

    def execute_min(self):
        return tf.minimum(self.a, self.b, name="minimum" + str(self.node_num))

    def execute_abs(self):
        return tf.abs(self.a, name="abs" + str(self.node_num))

    def execute_acos(self):
        return tf.acos(self.a, name="acos" + str(self.node_num))

    def execute_acosh(self):
        return tf.acosh(self.a, name="acosh" + str(self.node_num))

        '''
        tf.add_n([self.a, self.b], name="add_n" + str(self.node_num))
        tf.math_ops.betainc(self.a, self.b, self.x, name = "betainc" + str(self.node_num))
        tf.math_ops.conj(self.a, name="conj" + str(self.node_num))
        '''

    def execute_ceil(self):
        return tf.ceil(self.a, name="ceil" + str(self.node_num))

    def execute_cos(self):
        return tf.cos(self.a, name="cos" + str(self.node_num))

    def execute_cosh(self):
        return tf.cosh(self.a, name="cosh" + str(self.node_num))

    def execute_asin(self):
        return tf.asin(self.a, name="asin" + str(self.node_num))

    def execute_asinh(self):
        return tf.asinh(self.a, name="asinh" + str(self.node_num))

    def execute_atan(self):
        return tf.atan(self.a, name="atan" + str(self.node_num))

    def execute_atan2(self):
        return tf.atan2(self.a, self.b, name="atan2" + str(self.node_num))

    def execute_atanh(self):
        return tf.atanh(self.a, name="atanh" + str(self.node_num))

    def execute_count_nonzero(self):
        return tf.count_nonzero(self.a, name="count_nonzero" + str(self.node_num))

    def execute_cross(self):
        return tf.cross(self.a, self.b, name="cross" + str(self.node_num))

    def execute_cumprod(self):
        return tf.cumprod(self.a, name="cumprod" + str(self.node_num))

    def execute_cumsum(self):
        return tf.cumsum(self.a, name="cumsum" + str(self.node_num))

    def execute_exp(self):
        return tf.exp(self.a, name='exp' + str(self.node_num))

    def execute_log(self):
        return tf.log(self.a, name='log' + str(self.node_num))

    def execute_log1p(self):
        return tf.log1p(self.a, name='log1p' + str(self.node_num))

    def execute_mod(self):
        return tf.mod(self.a, self.b, name='mode' + str(self.node_num))

    def execute_mathmul(self):
        return tf.matmul(self.a, self.b, name="matmul" + str(self.node_num))

    def execute_erf(self):
        return tf.erf(self.a, name='erf' + str(self.node_num))

    def execute_diag(self):
        return tf.diag(self.a, name="diag" + str(self.node_num))

    def execute_diag_part(self):
        return tf.diag_part(self.a, name="diag_part" + str(self.node_num))

    def execute_elu(self):
        return tf.nn.elu(self.a, name="elu" + str(self.node_num))

    def execute_expm(self):
        return tf.expm1(self.a, name="expm" + str(self.node_num))

    def execute_floor(self):
        return tf.floor(self.a, name="floor" + str(self.node_num))

    def execute_sin(self):
        return tf.sin(self.a, name="sin" + str(self.node_num))

    def execute_sinh(self):
        return tf.sinh(self.a, name="sinh" + str(self.node_num))

    def execute_mean(self):
        return tf.metrics.mean(self.a, self.b, name="mean" + str(self.node_num))
