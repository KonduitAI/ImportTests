import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph
from tfoptests.math_ops import DifferentiableMathOps


class MathOpsTwo(TestGraph):
    def __init__(self, *args, **kwargs):
        super(MathOpsTwo, self).__init__(*args, **kwargs)
        self.input_0 = np.ndarray(shape=(1,3), dtype=int)

    def list_inputs(self):
        return ["input_0"]

    def get_placeholder_input(self, name):
        if name == "input_0":
            return self.input_0

    def _get_placeholder_shape(self, name):
        if name == "input_0" or name == "input_1":
            return [1, 3]


def test_mathops_two():
    mathops_2 = MathOpsTwo(seed=19)
    in_node_0 = mathops_2.get_placeholder("input_0", data_type=tf.int32)

    arrs = []
    for i in range(1, 5, 1):
        arrs.append(tf.Variable(tf.constant(5, dtype=tf.int32, shape=(1, 1))))
    out_node = tf.math.bincount(arrs)

    placeholders = [in_node_0]
    predictions = [out_node]
    # Run and persist
    tfp = TensorFlowPersistor(save_dir="test_bincount")
    tfp.set_placeholders([]) \
        .set_output_tensors(predictions) \
        .set_test_data(mathops_2.get_test_data()) \
        .build_save_frozen_graph()


if __name__ == '__main__':
    test_mathops_two()
