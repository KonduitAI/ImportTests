import urllib
from io import BytesIO

from PIL import Image
from keras import Model
from keras.applications import InceptionResNetV2
from keras.applications.inception_resnet_v2 import decode_predictions
from keras.engine.saving import load_model
from keras.initializers import glorot_uniform
import keras.backend as K
import tensorflow as tf
from keras_applications.inception_resnet_v2 import preprocess_input
import numpy as np

from model_zoo.util import inception_preprocessing
from model_zoo.util.freeze_keras import freeze_keras_model

tf.keras.backend.set_learning_phase(0)
K.set_learning_phase(0)

model: Model = InceptionResNetV2()

K.set_image_data_format("channels_last")

f = urllib.request.urlopen("https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/img/image2.jpg?raw=true")
jpeg_str = f.read()
image = Image.open(BytesIO(jpeg_str))
image = np.asarray(image.convert('RGB'))
image = tf.convert_to_tensor(image)
x = inception_preprocessing.preprocess_for_eval(image, 299, 299)
x = tf.Session().run(x)
x = np.expand_dims(x, axis=0)
print(np.shape(x))
preds = model.predict([x,])
print('Predicted:', decode_predictions(preds, top=3)[0])

for layer in model.layers:
    layer.training = False
    layer.trainable = False
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer._per_input_updates = {}
    elif isinstance(layer, tf.keras.layers.Dropout):
        layer._per_input_updates = {}

freeze_keras_model(model, "C:\\Temp\\TF_Graphs\\" + "keras_InceptionResNetV2")
