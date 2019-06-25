import numpy as np
import tensorflow as tf

def preprocess_for_eval(image, height, width):
    i = tf.convert_to_tensor(image)
    image = tf.Session().run(tf.image.convert_image_dtype(i, dtype=tf.float32))
    image = np.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [height, width],
                                     align_corners=False)
    image = tf.squeeze(image, axis=0)
    return image