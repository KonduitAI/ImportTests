import hashlib
import os
import shutil
import traceback
from io import BytesIO
from pathlib import Path
from shutil import copyfile

import tensorflow as tf
import numpy as np
import typing
from six.moves import urllib
from PIL import Image  # pip install Pillow

from tfoptests.persistor import TensorFlowPersistor
from model_zoo.util import vgg_preprocessing
from model_zoo.util import inception_preprocessing
from model_zoo.util import hub_processing

key = os.getenv("AZURE_KEY", None)


class ZooEvaluation(object):

    def __init__(self, name,
                 baseDir="/dl4j-test-resources/src/main/resources/tf_graphs"
                         "/zoo_models",
                 prefix="graph"):
        tf.reset_default_graph()
        self.name = name
        self.baseDir = baseDir
        self.prefix = prefix
        self.image_url = None
        self.save_graph = False
        self.is_image = False

    def graphFile(self, graphFile):
        self.graphFile = graphFile
        return self

    def outputDir(self, outputDir):
        self.outputDir = outputDir
        return self

    def imageUrl(self, imageUrl):
        self.image_url = imageUrl
        self.is_image = True
        return self

    def inputDims(self, h, w, c):
        self.inputH = h
        self.inputW = w
        self.inputC = c
        self.is_image = True
        return self

    def saveGraph(self):
        self.save_graph = True
        return self

    def inputNames(self, inputNames):
        self.input_names = inputNames
        return self

    def inputName(self, inputName):
        return self.inputNames(inputName)

    def outputNames(self, outputNames):
        self.outputNames = outputNames
        return self

    def preprocessingType(self, preprocessingType):
        self.preprocessingType = preprocessingType
        return self

    def setData(self, data):
        self.data = data
        return self

    def setSingleBatchData(self, data):
        self.setData(np.expand_dims(data, 0))
        return self

    def getImage(self, expandDim):
        if (self.inputH is None):
            raise ValueError("input height not set")
        if (self.inputC is not 3):
            raise ValueError("Only ch=3 implemented so far")
        if (self.image_url is None):
            raise ValueError("Image URL is not set")
        if (self.inputH != self.inputW):
            raise ValueError("Non-square inputs not yet implemented")

        try:
            f = urllib.request.urlopen(self.image_url)
            jpeg_str = f.read()
            image = Image.open(BytesIO(jpeg_str))
        except IOError:
            raise ValueError(
                'Cannot retrieve image. Please check url: ' + self.image_url)

        image = np.asarray(image.convert('RGB'))
        print("image shape: ", image.shape)
        m = min(image.shape[0], image.shape[1])
        print("min: ", m)
        if (self.preprocessingType == "vgg"):
            image = vgg_preprocessing.preprocess_for_eval(image, self.inputH,
                                                          self.inputW, m)
        elif (self.preprocessingType == "inception"):
            # image = image.astype("float32")
            # with tf.Session() as sess:
            #     image = image.eval(session=sess)
            image = tf.convert_to_tensor(image)
            image = inception_preprocessing.preprocess_for_eval(image,
                                                                height=self.inputH,
                                                                width=self.inputW)
        elif (self.preprocessingType == "resize_only"):
            image = np.expand_dims(image, 0)
            image = tf.image.resize_bilinear(image, [self.inputH, self.inputW],
                                             align_corners=False)
            image = tf.squeeze(image, axis=0)
            # image = tf.convert_to_tensor(image)
        elif (self.preprocessingType == "hub"):
            image = hub_processing.preprocess_for_eval(image, self.inputH,
                                                       self.inputW)
        else:
            raise ValueError("Unknown preprocessing type: ",
                             self.preprocessingType)
        # image = tf.expand_dims(image, 0)
        print("new shape: ", image.shape)
        print("new type: ", type(image))

        with tf.Session() as sess:
            image = image.eval(session=sess)
        # print("new shape2: ", image.shape)
        # print("new type2: ", type(image))

        if (len(image.shape) == 3 and expandDim):
            image = np.expand_dims(image, 0)

        print("new shape3: ", image.shape)
        print("new type3: ", type(image))

        return image

        # width, height = image.size
        # #To resize keeping aspect ratio:
        # #resize_ratio = 1.0 * self.inputH / max(width, height)
        # #target_size = (int(resize_ratio * width), int(resize_ratio * height))
        # target_size = [self.inputH, self.inputW]
        # # print("Target size: ", target_size)
        # resized_image = image.convert('RGB').resize(target_size,
        # Image.ANTIALIAS)
        # input = np.asarray(resized_image)
        # return input

    def get_feed_dict(self):
        if (self.input_names is None or self.outputNames is None):
            raise ValueError("inputNames or outputNames not set")

        if self.is_image:
            data = self.getImage(False)
        else:
            data = self.data

        if isinstance(data, dict):
            if not isinstance(self.input_names, list):
                if len(data) != 1:
                    raise ValueError("Given multiple input datas for one input")
                else:
                    data = data[list(data.keys())[0]]
            else:
                if set(data.keys()) != set(self.input_names):
                    raise ValueError("Input names and given inputs don't match")

        if (data is None):
            raise ValueError("Null input data")

        print("Input names: ", self.input_names)
        print("Output names: ", self.outputNames)
        print("Input data shape: ", data.shape)
        if isinstance(data, dict):
            feed_dict = {k: [v] for k, v in data.items()}
        else:
            if self.is_image:
                feed_dict = {self.input_names: [data]}
            else:
                feed_dict = {self.input_names: data}

        return feed_dict, data

    # for use with org.nd4j.imports.listeners.ImportModelDebugger
    def write_intermediates(self, dest):

        if not isinstance(dest, Path):
            dest = Path(dest)

        feed_dict, data = self.get_feed_dict()

          # Save input
        for inName in feed_dict:
            inPath = dest / "__placeholders" / (inName.replace(":", "__") + ".npy")
            inPath = inPath.absolute()
            inPath.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(inPath), feed_dict[inName])
        

        graph = self.loadGraph()

          # now build the graph
        with graph.as_default() as graph:
            for op in graph.get_operations():
                print(op.name)
                  # print("  ", op.outputs)
                sess = tf.Session()
                nOuts = len(op.outputs)
                for i in range(nOuts):
                    try:
                        out = sess.run([op.name + ":" + str(i)], feed_dict=feed_dict)

                        path = dest
                        for p in op.name.split("/")[:-1]:
                            path = path / p

                        path: Path = path / (op.name.split("/")[-1] + "__" + str(i) + ".npy")
                        path = path.absolute()
                        path.parent.mkdir(parents=True, exist_ok=True)

                        np.save(str(path), out[0])
                    except Exception:
                        print("Error saving " + op.name + ":" + str(i))
                        traceback.print_exc()
                        print("-------------------------------------------------------------")

    def write(self):

        graph = self.loadGraph()

        with tf.Session(graph=graph) as sess:

            feed_dict, data = self.get_feed_dict()

            outputs = sess.run(
                # self.outputName,
                self.outputNames,
                feed_dict=feed_dict)
            # print(outputs)

        # print("Outputs: ", outputs)

        toSave = {}
        toSave_dtype_dict = {}
        for i in range(len(outputs)):
            toSave[self.outputNames[i]] = outputs[i]
            print("Output: ", self.outputNames[i])
            print("Dtype: ", outputs[i].dtype)
            toSave_dtype_dict[self.outputNames[i]] = str(outputs[i].dtype)

        # print("Values to save: ", toSave)
        filename = self.name + "_frozenmodel.pb"

        # remove old files
        if os.path.exists(self.baseDir + "/" + self.name + "/"):
            shutil.rmtree(self.baseDir + "/" + self.name + "/")

        tfp = TensorFlowPersistor(base_dir=self.baseDir, save_dir=self.name,
                                  verbose=False)

        if self.save_graph:
            copyfile(self.graphFile,
                     self.baseDir + "/" + self.name + "/" + filename)

            if key is not None:
                command = f"az storage blob upload --file {filename} " \
                          f"--account-name deeplearning4jblob " + \
                          f"--account-key {key} " + \
                          f"--container-name testresources --name {filename}"
            else:
                command = f"az storage blob upload --file {filename} " \
                          f"--account-name deeplearning4jblob " + \
                          f"--container-name testresources --name {filename}"

            print(command)

            with open(self.baseDir + "/" + self.name + "/" + filename,
                      'rb') as f:
                md5 = hashlib.md5(f.read()).hexdigest()
            with open(self.baseDir + "/" + self.name + "/tf_model.txt",
                      'w+') as f:
                f.write(
                    f"https://deeplearning4jblob.blob.core.windows.net"
                    f"/testresources/{filename}\n" + md5 + "\n" + filename)

        if self.is_image:
            tfp._save_input(self.getImage(True), self.input_names)
        else:
            if isinstance(self.data, dict):
                for k, v in self.data.items():
                    tfp._save_input(v, k)
            else:
                tfp._save_input(data, self.input_names)

        dtype_dict = {}
        if isinstance(data, dict):
            for k, v in data:
                dtype_dict[k] = str(v.dtype)
        else:
            dtype_dict[self.input_names] = str(data.dtype)

        tfp._save_node_dtypes(dtype_dict)
        # tfp._save_predictions({self.outputName:outputs})
        tfp._save_predictions(toSave)
        tfp._save_node_dtypes(toSave_dtype_dict)

        # Also save intermediate nodes:
        # dict = {self.inputName:image}
        # dict = {self.inputName:self.getImage(True)}
        # print("DICTIONARY: ", dict)
        # print("OUTPUT NAMES: ", self.outputNames)
        # tfp.set_output_tensors(self.outputNames)
        # tfp.set_verbose(True)
        # tfp._save_intermediate_nodes2(input_dict=dict, graph=graph)

    def loadGraph(self):
        if (self.graphFile is None):
            raise ValueError("Graph file location not set")

        with tf.gfile.GFile(self.graphFile, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # Then, we import the graph_def into a new Graph and returns it
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name=self.prefix)
        return graph


if __name__ == '__main__':
    # How to run this?
    # 1. Download the required models, links below
    # 2. For many models, just implement using the ZooEvaluation class.
    # 3. Some models need special treatment - see eval_data directory for
    # these cases
    # 4. Run with: python model_zoo/util/zoo_evaluation.py
    # See also: https://gist.github.com/eraly/7d48807ed2c69233072ed06c12bf9b0a

    # DenseNet - uses vgg preprocessing according to readme
    # https://storage.googleapis.com/download.tensorflow.org/models/tflite
    # /model_zoo/upload_20180427/densenet_2018_04_27.tgz
    # z = ZooEvaluation(name="densenet_2018_04_27",prefix="")
    # z.graphFile("C:\\Temp\\TF_Graphs\\densenet_2018_04_27\\densenet.pb")\
    #     .inputName("Placeholder:0")\
    #     .outputNames(["ArgMax:0", "softmax_tensor:0"])\
    #     .imageUrl("https://github.com/tensorflow/models/blob/master
    #     /research/deeplab/g3doc/img/image2.jpg?raw=true")\
    #     .inputDims(224, 224, 3)\
    #     .preprocessingType("vgg")
    # z.write()
    # # SqueezeNet: also vgg preprocessing
    # # https://storage.googleapis.com/download.tensorflow.org/models/tflite
    # /model_zoo/upload_20180427/squeezenet_2018_04_27.tgz
    # z = ZooEvaluation(name="squeezenet_2018_04_27",prefix="")
    # z.graphFile("C:\\Temp\\TF_Graphs\\squeezenet_2018_04_27\\squeezenet.pb")\
    #     .inputName("Placeholder:0") \
    #     .outputNames(["ArgMax:0", "softmax_tensor:0"])\
    #     .imageUrl("https://github.com/tensorflow/models/blob/master
    #     /research/deeplab/g3doc/img/image2.jpg?raw=true")\
    #     .inputDims(224, 224, 3)\
    #     .preprocessingType("vgg")
    # z.write()
    #
    # nasnet_mobile: no preprocessing specified, going to assume inception
    # preprocessing
    # https://storage.googleapis.com/download.tensorflow.org/models/tflite
    # /model_zoo/upload_20180427/nasnet_mobile_2018_04_27.tgz
    # z = ZooEvaluation(name="nasnet_mobile_2018_04_27",prefix="")
    # z.graphFile("C:\\Temp\\TF_Graphs\\nasnet_mobile_2018_04_27
    # \\nasnet_mobile.pb") \
    #     .inputName("input:0") \
    #     .outputNames(["final_layer/predictions:0"]) \
    #     .imageUrl("https://github.com/tensorflow/models/blob/master
    #     /research/deeplab/g3doc/img/image2.jpg?raw=true") \
    #     .inputDims(224, 224, 3) \
    #     .preprocessingType("inception")
    # z.write()
    #
    # # https://storage.googleapis.com/download.tensorflow.org/models/tflite
    # /model_zoo/upload_20180427/inception_v4_2018_04_27.tgz
    # z = ZooEvaluation(name="inception_v4_2018_04_27",prefix="")
    # z.graphFile("C:\\Temp\\TF_Graphs\\inception_v4_2018_04_27\\inception_v4
    # .pb") \
    #     .inputName("input:0") \
    #     .outputNames(["InceptionV4/Logits/Predictions:0"]) \
    #     .imageUrl("https://github.com/tensorflow/models/blob/master
    #     /research/deeplab/g3doc/img/image2.jpg?raw=true") \
    #     .inputDims(299, 299, 3) \
    #     .preprocessingType("inception")
    # z.write()
    #
    # # https://storage.googleapis.com/download.tensorflow.org/models/tflite
    # /model_zoo/upload_20180427/inception_resnet_v2_2018_04_27.tgz
    # z = ZooEvaluation(name="inception_resnet_v2_2018_04_27",prefix="")
    # z.graphFile("C:\\Temp\\TF_Graphs\\inception_resnet_v2_2018_04_27
    # \\inception_resnet_v2.pb") \
    #     .inputName("input:0") \
    #     .outputNames(["InceptionResnetV2/AuxLogits/Logits/BiasAdd:0"]) \
    #     .imageUrl("https://github.com/tensorflow/models/blob/master
    #     /research/deeplab/g3doc/img/image2.jpg?raw=true") \
    #     .inputDims(299, 299, 3) \
    #     .preprocessingType("inception")
    # z.write()
    #
    # # http://download.tensorflow.org/models/mobilenet_v1_2018_02_22
    # /mobilenet_v1_0.5_128.tgz
    # z = ZooEvaluation(name="mobilenet_v1_0.5_128",prefix="")   #None)
    # #"mobilenet_v1_0.5")
    # z.graphFile("C:\\Temp\\TF_Graphs\\mobilenet_v1_0.5_128\\mobilenet_v1_0
    # .5_128_frozen.pb") \
    # .inputName("input:0") \
    # .outputNames(["MobilenetV1/Predictions/Reshape_1:0"]) \
    # .imageUrl("https://github.com/tensorflow/models/blob/master/research
    # /deeplab/g3doc/img/image2.jpg?raw=true") \
    # .inputDims(128, 128, 3) \
    # .preprocessingType("inception")     #Not 100% sure on this, but more
    # likely it's inception than vgg preprocessing...
    # z.write()
    #
    # # http://download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1
    # .0_224.tgz
    # z = ZooEvaluation(name="mobilenet_v2_1.0_224",prefix="")
    # z.graphFile("C:\\Temp\\TF_Graphs\\mobilenet_v2_1.0_224\\mobilenet_v2_1
    # .0_224_frozen.pb") \
    #     .inputName("input:0") \
    #     .outputNames(["MobilenetV2/Predictions/Reshape_1:0"]) \
    #     .imageUrl("https://github.com/tensorflow/models/blob/master
    #     /research/deeplab/g3doc/img/image2.jpg?raw=true") \
    #     .inputDims(224, 224, 3) \
    #     .preprocessingType("inception")     #Not 100% sure on this,
    #     but more likely it's inception than vgg preprocessing...
    # z.write()
    #
    # http://download.tensorflow.org/models/official
    # /resnetv2_imagenet_frozen_graph.pb
    # https://github.com/tensorflow/models/blob/master/research/tensorrt
    # /README.md
    # "Some ResNet models represent 1000 categories, and some represent all
    # 1001, with the 0th category being "background". The models provided are
    # of the latter type."
    # Runs "imagenet_preprocessing" on the images -
    # https://github.com/tensorflow/models/blob/master/official/resnet
    # /imagenet_preprocessing.py
    # Which seems to be merely scaling, no normalization
    # z = ZooEvaluation(name="resnetv2_imagenet_frozen_graph",prefix="")
    # z.graphFile("/TF_Graphs/resnetv2_imagenet_frozen_graph.pb") \
    #     .inputName("input_tensor:0") \
    #     .outputNames(["softmax_tensor:0"]) \
    #     .imageUrl("https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/img/image2.jpg?raw=true") \
    #     .inputDims(224, 224, 3) \
    #     .preprocessingType("resize_only")
    # z.write()

    # # http://download.tensorflow.org/models/object_detection
    # /ssd_mobilenet_v1_coco_2018_01_28.tar.gz
    # # https://github.com/tensorflow/models/blob/master/research
    # /object_detection/object_detection_tutorial.ipynb
    # # Seems like this can be use on (nearly) any image size??? Docs and
    # notebook are very vague on this point
    # # 320x320 input is arbitrary, not based on anything
    # # Outputs: detection_boxes, detection_scores, num_detections,
    # detection_classes
    # # Preprocessing: unclear from docs, but it does have some preprocessing
    # built into the network
    # z = ZooEvaluation(name="ssd_mobilenet_v1_coco_2018_01_28",prefix="")
    # z.graphFile("C:\\Temp\\TF_Graphs\\ssd_mobilenet_v1_coco_2018_01_28
    # \\frozen_inference_graph.pb") \
    #     .inputName("image_tensor:0") \
    #     .outputNames(["detection_boxes:0", "detection_scores:0",
    #     "num_detections:0", "detection_classes:0"]) \
    #     .imageUrl("https://github.com/tensorflow/models/blob/master
    #     /research/deeplab/g3doc/img/image2.jpg?raw=true") \
    #     .inputDims(320, 320, 3) \
    #     .preprocessingType("resize_only")     #Not 100% sure on this,
    #     but seems most likely
    # z.write()

    # http://download.tensorflow.org/models/object_detection
    # /ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03.tar.gz
    # As above, for ssd_mobilenet_v1_coco_2018_01_28
    # z = ZooEvaluation(
    # name="ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03",
    # prefix="")
    # z.graphFile("C:\\Temp\\TF_Graphs\\ssd_mobilenet_v1_0
    # .75_depth_300x300_coco14_sync_2018_07_03\\frozen_inference_graph.pb") \
    #     .inputName("image_tensor:0") \
    #     .outputNames(["detection_boxes:0", "detection_scores:0",
    #     "num_detections:0", "detection_classes:0"]) \
    #     .imageUrl("https://github.com/tensorflow/models/blob/master
    #     /research/deeplab/g3doc/img/image2.jpg?raw=true") \
    #     .inputDims(300, 300, 3) \
    #     .preprocessingType("resize_only")     #Not 100% sure on this,
    #     but seems most likely
    # z.write()

    # # http://download.tensorflow.org/models/object_detection
    # # /faster_rcnn_resnet101_coco_2018_01_28.tar.gz
    # # https://github.com/tensorflow/models/blob/master/research
    # # /object_detection/g3doc/detection_model_zoo.md
    # # Runtimes are reported for this model on 600x600 images, so use that as
    # # it's definitely supported
    # z = ZooEvaluation(name="faster_rcnn_resnet101_coco_2018_01_28", prefix="")
    # # z.graphFile("C:\\Temp\\TF_Graphs\\faster_rcnn_resnet101_coco_2018_01_28
    # # \\frozen_inference_graph.pb") \
    # z.graphFile(
    #     "/TF_Graphs/faster_rcnn_resnet101_coco_2018_01_28"
    #     "/frozen_inference_graph.pb") \
    #     .inputName("image_tensor:0") \
    #     .outputNames(
    #     ["detection_boxes:0", "detection_scores:0", "num_detections:0",
    #      "detection_classes:0"]) \
    #     .imageUrl(
    #     "https://github.com/tensorflow/models/blob/master/research/deeplab"
    #     "/g3doc/img/image2.jpg?raw=true") \
    #     .inputDims(600, 600, 3) \
    #     .preprocessingType("resize_only")  # Should be this, given it has a
    # # crop and resize op internally? Not 100% sure on normalization
    # z.write()

    # Author prediction RNN I had laying around
    # z = ZooEvaluation(name="PorV-RNN", prefix="")
    # z.graphFile("/TF_Graphs/PorVRNN/tf_model.pb") \
    #     .inputName("input_1:0") \
    #     .outputNames(["dense_2/Sigmoid:0"]) \
    #     .setSingleBatchData(np.array(
    #     [0, 0, 0, 0, 0, 3, 39, 9, 342, 8519, 1, 2768, 6022, 1777, 1, 155, 8,
    #      490, 1, 202, 4, 1, 2768, 23, 34, 1, 2768, 8520, 2518, 58, 1, 6022,
    #      3101, 13, 3, 1, 155, 8, 46, 2161])) \
    #     .saveGraph()
    #
    # z.write()

    # Text generation RNN
    # https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/8.1-text-generation-with-lstm.ipynb

    # seed text: this was a
    # start = np.zeros((60,59))
    # start[0, 46] = 1
    # start[1, 34] = 1
    # start[2, 35] = 1
    # start[3, 45] = 1
    # start[4, 1] = 1
    # start[5, 49] = 1
    # start[6, 27] = 1
    # start[7, 45] = 1
    # start[8, 1] = 1
    # start[9, 27] = 1


    # z = ZooEvaluation(name="text_gen_81", prefix="")
    # z.graphFile("/TF_Graphs/text_gen_81/tf_model.pb") \
    #     .inputName("lstm_1_input:0") \
    #     .outputNames(["dense_1_1/Softmax:0"]) \
    #     .setSingleBatchData(start) \
    #     .saveGraph()
    #
    # z.write()

    # # CIFAR-10 DCGAN (just the generator)
    # # https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/8.5-introduction-to-gans.ipynb
    # z = ZooEvaluation(name="cifar10_gan_85", prefix="")
    # z.graphFile("/TF_Graphs/cifar10_gan_85/tf_model.pb") \
    #     .inputName("input_1:0") \
    #     .outputNames(['conv2d_4/Tanh:0']) \
    #     .setSingleBatchData(
    #     np.array([-0.66637576, -0.75804161, 0.70265126, 0.67644233, 1.70802486,
    #               0.67723204, -0.95535933, 0.4106528, 0.15204615, 0.81687495,
    #               0.07579885, -0.84215164, 1.4546437, 0.73752796, -0.68664101,
    #               0.02874679, -0.26094784, -1.20962916, 0.38224013, -0.76390132,
    #               0.9095366, 0.85208794, -0.80977076, 1.91582847, 2.87287804,
    #               2.35497939, 0.34249151, -0.1988978, -0.07104926, 1.83731291,
    #               0.70314201, 0.33953821])) \
    #     .saveGraph()

    # z.write()
    # z.write_intermediates("/TF_Graphs/cifar10_gan_85/")

    # tempData = np.array([[1.4307807, -1.08526969, -1.18351695, -0.79942328, 1.15056955,
    #                -0.97578531, -0.89015785, -0.77802672, -0.90604984,
    #                -0.906668,
    #                1.38567444, -0.54078408, -0.3473285, 0.06561315],
    #               [1.36356716, -1.20839876, -1.29967599, -0.92083247,
    #                1.24024323,
    #                -1.0356099, -0.97352926, -0.80075238, -0.9848299,
    #                -0.98662918,
    #                1.49276138, -0.57332209, -0.77638912, 0.22840583],
    #               [1.27041155, -1.31684272, -1.40131515, -1.02549556,
    #                1.33589515,
    #                -1.08372968, -1.04022639, -0.81934609, -1.05235566,
    #                -1.05247956,
    #                1.57984306, 0.06442285, -0.12421698, 0.12795928],
    #               [1.19966045, -1.31006497, -1.3890291, -1.01014497, 1.35382989,
    #                -1.08112861, -1.03069823, -0.82347803, -1.04110137,
    #                -1.04307236,
    #                1.55465998, -0.35206364, -0.3473285, 0.20300555],
    #               [1.12655099, -1.26826886, -1.3410018, -0.96967524, 1.31796042,
    #                -1.06162059, -1.00449578, -0.81521416, -1.01484135,
    #                -1.01485077,
    #                1.49229067, 0.16203687, 0.01737303, 0.0448311]])

    # # Temperature prediction w/ stacked GRUs
    # # https://github.com/fchollet/deep-learning-with-python-notebooks/blob
    # # /master/6.3-advanced-usage-of-recurrent-neural-networks.ipynb
    # z = ZooEvaluation(name="temperature_stacked_63", prefix="")
    # z.graphFile("/TF_Graphs/temperature_stacked_63/tf_model.pb") \
    #     .inputName("gru_1_input:0") \
    #     .outputNames(['dense_1_1/BiasAdd:0']) \
    #     .setSingleBatchData(tempData) \
    #     .saveGraph()
    #
    # z.write()


    # # Temperature prediction w/ bidirectional GRU
    # # https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/6.3-advanced-usage-of-recurrent-neural-networks.ipynb
    # z = ZooEvaluation(name="temperature_bidirectional_63", prefix="")
    # z.graphFile("/TF_Graphs/temperature_bidirectional_63/tf_model.pb") \
    #     .inputName("bidirectional_1_input:0") \
    #     .outputNames(['dense_1/BiasAdd:0']) \
    #     .setSingleBatchData(tempData) \
    #     .saveGraph()
    #
    # z.write()



    #TODO does not work, see https://github.com/keras-team/keras/issues/12588

    # Temperature prediction w/ 1D CNN and a GRU
    # https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/6.3-advanced-usage-of-recurrent-neural-networks.ipynb
    # z = ZooEvaluation(name="temperature_1dcnn_to_gru_64", prefix="")
    # z.graphFile("/TF_Graphs/temperature_1dcnn_to_gru_64/tf_model.pb") \
    #     .inputName("conv1d_1_input:0") \
    #     .outputNames(['dense_1/BiasAdd:0']) \
    #     .setSingleBatchData(tempData) \
    #     .saveGraph()
    #
    # z.write()

    # Image compression done with a residual gru
    #  https://github.com/tensorflow/models/tree/master/research/compression/image_encoder
    # download.tensorflow.org\models\compression_residual_gru-2016-08-23.tar.gz
    # z = ZooEvaluation(name="compression_residual_gru",prefix="")
    # z.graphFile("/TF_Graphs/compression_residual_gru/residual_gru.pb") \
    #     .inputName("Placeholder:0") \
    #     .outputNames(["GruBinarizer/SignBinarizer/Sign:0"]) \
    #     .imageUrl("https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/img/image2.jpg?raw=true") \
    #     .inputDims(320, 320, 3) \
    #     .preprocessingType("resize_only")
    #
    # z.write()


    # # Image semantic segmentation
    # # https://github.com/tensorflow/models/tree/master/research/deeplab
    # # download.tensorflow.org\models\deeplabv3_xception_ade20k_train_2018_05_29.tar.gz
    # z = ZooEvaluation(name="deeplabv3_xception_ade20k_train",prefix="")
    # z.graphFile("/TF_Graphs/deeplabv3_xception_ade20k_train/frozen_inference_graph.pb") \
    #     .inputName("ImageTensor:0") \
    #     .outputNames(["SemanticPredictions:0"]) \
    #     .imageUrl("https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/img/image2.jpg?raw=true") \
    #     .inputDims(300, 300, 3) \
    #     .preprocessingType("resize_only")
    #
    # z.write()

    # # Alexnet
    # # jaina.cs.ucdavis.edu\datasets\adv\imagenet\alexnet_frozen.pb
    # z = ZooEvaluation(name="alexnet", prefix="")
    # z.graphFile("/TF_Graphs/alexnet/alexnet_frozen.pb") \
    #     .inputName("Placeholder:0") \
    #     .outputNames(["Softmax:0"]) \
    #     .imageUrl(
    #     "https://github.com/tensorflow/models/blob/master/research/deeplab"
    #     "/g3doc/img/image2.jpg?raw=true") \
    #     .inputDims(227, 227, 3) \
    #     .preprocessingType("resize_only")
    #
    # z.write()


    # graph = z.loadGraph()
    #
    # for op in graph.get_operations():
    #     print(op.name)
