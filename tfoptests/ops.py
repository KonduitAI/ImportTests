import tensorflow as tf
import numpy as np


class OpCreator:
    def __init__(self, op):
        self.op = op
        self.node_num = 0

    def setVars(self, vars):
        self.vars = vars

    def setPlaceholders(self, placeholders):
        self.placeholders = placeholders

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

    def execute_segment_max(self):
        return [tf.segment_max(data=self.vars[0], segment_ids=self.vars[1])]

    def execute_segment_min(self):
        return [tf.segment_min(data=self.vars[0], segment_ids=self.vars[1])]

    def execute_segment_mean(self):
        return [tf.segment_mean(data=self.vars[0], segment_ids=self.vars[1])]

    def execute_segment_prod(self):
        return [tf.segment_prod(data=self.vars[0], segment_ids=self.vars[1])]

    def execute_segment_sum(self):
        return [tf.segment_sum(data=self.vars[0], segment_ids=self.vars[1])]

    def execute_space_to_batch(self):
        return [tf.space_to_batch(input=self.vars[0], paddings=self.vars[1], block_size=2)]

    def execute_space_to_depth(self):
        return [tf.space_to_depth(input=self.vars[0], block_size=2, data_format=self.op["data_format"])]

    def execute_batch_to_space(self):
        return [tf.batch_to_space(input=self.vars[0], crops=self.vars[1], block_size=2)]

    def execute_depth_to_space(self):
        return [tf.depth_to_space(input=self.vars[0], block_size=2, data_format=self.op["data_format"])]

    def execute_size(self):
        temp = tf.add(self.vars[0], 1.0)
        return [tf.add(tf.size(input=temp), 1)]

    def execute_shape(self):
        temp = tf.add(self.vars[0], 1.0)
        return [tf.add(tf.shape(input=temp), 1)]

    def execute_shapen(self):
        out = tf.shape_n(input=self.vars)
        #Concat multiple outputs to avoid graph saving issue
        return [tf.concat(out, axis=0)]

    def execute_matrix_inverse(self):
        return [tf.matrix_inverse(input=self.vars[0])]

    def execute_pad(self):
        if(len(self.vars) > 2):
            return [tf.pad(tensor=self.vars[0], paddings=self.vars[1], constant_values=self.vars[2], mode = self.op["mode"])]
        else:
            return [tf.pad(tensor=self.vars[0], paddings=self.vars[1], mode=self.op["mode"])]

    def execute_unique(self):
        #Hack for multi-output saving issue: concat
        temp = tf.unique(self.vars[0])
        toConcat = []
        toConcat.append(temp[0])
        toConcat.append(tf.cast(temp[1], dtype=tf.float32))
        return [tf.concat(toConcat, axis=0)]

    def execute_unique_with_counts(self):
        temp = tf.unique_with_counts(self.vars[0])
        toConcat = []
        toConcat.append(temp[0])
        toConcat.append(tf.cast(temp[1], dtype=tf.float32))
        toConcat.append(tf.cast(temp[2], dtype=tf.float32))
        return [tf.concat(toConcat,axis=0)]

    def execute_topk(self):
        temp = tf.nn.top_k(input=self.vars[0], k=self.op["k"], sorted=self.op["sorted"])
        toConcat = []
        toConcat.append(temp[0])
        toConcat.append(tf.cast(temp[1], dtype=tf.float32))
        #Concat multiple outputs to avoid graph saving issue. Note that values and indices have same shape
        return [tf.concat(toConcat, axis=0)]

    def execute_in_top_k(self):
        return [tf.nn.in_top_k(predictions=self.vars[0], targets=self.vars[1], k=self.op["k"])]

    def execute_matrix_determinant(self):
        return [tf.matrix_determinant(input=self.vars[0])]

    def execute_matrix_set_diag(self):
        return [tf.matrix_set_diag(input=self.vars[0], diagonal=self.vars[1])]

    def execute_identity_n(self):
        return tf.identity_n(self.vars)

    def execute_zeta(self):
        x = tf.add(self.vars[0], 1.0)    #x values must be > 1
        return [tf.zeta(x=x, q=self.vars[1])]

    def execute_confusion_matrix(self):
        weights = None
        if(len(self.vars) > 2):
            weights = self.vars[2]
        return [tf.confusion_matrix(labels=self.vars[0], predictions=self.vars[1], num_classes=self.op["num_classes"], weights=weights)]

    def execute_stack(self):
        return [tf.stack(values=self.vars, axis=self.op["axis"])]

    def execute_parallel_stack(self):
        return [tf.parallel_stack(values=self.vars)]

    def execute_accumulate_n(self):
        return [tf.accumulate_n(self.vars)]

    def execute_angle(self):
        return [tf.add(tf.angle(self.vars[0]), 1.0)]

    def execute_approximate_equal(self):
        return [tf.approximate_equal(self.vars[0], self.vars[1])]

    def execute_matmul(self):
        ta = self.op.get("transpose_a", False)
        tb = self.op.get("transpose_b", False)
        print(self.op)
        print("ta = ",ta)
        print("tb = ",tb)
        return [tf.matmul(self.vars[0], self.vars[1], transpose_a=ta, transpose_b=tb, name = "matmul-" + str(self.node_num))]

    def execute_matrix_diag_part(self):
        return [tf.matrix_diag_part(self.vars[0])]

    def execute_svd(self):
        shapes = self.op["varShapes"]
        if(shapes[len(shapes)-1] != shapes[len(shapes)-2]):
            raise ValueError("Only square inputs currently supported due to multiple outputs issue")

        svd = tf.svd(tensor=self.vars[0], full_matrices=self.op["full_matrices"], compute_uv=self.op["compute_uv"])
        #Outputs: If compute_uv is false, only one output
        if(self.op["compute_uv"] is False or len(svd) == 1):
            if(isinstance(svd, list)):
                return svd
            return [svd]

        #Multiple outputs issue: s, shape [..., P], u shape [..., M, P] or [..., M, M]
        # v shape [..., N,P] or [..., N, N]
        # Where P is min(M,N)
        #Workaround for multiple outputs saving issue: if m=n, can add u and v... then need to reshape s to [..., M, 1] and broadcast add...
        s = svd[0]
        u = svd[1]
        v = svd[2]
        if(self.op["full_matrices"] is False):
            s = tf.expand_dims(s, -1)
        return [s + u + v]

    def execute_mean_squared_error(self):
        weights = 1.0
        if(len(self.vars) > 2):
            weights = self.vars[2]

        return [tf.losses.mean_squared_error(labels=self.vars[0], predictions=self.vars[1], weights=weights)]

    def execute_absolute_difference(self):
        weights = 1.0
        if(len(self.vars) > 2):
            weights = self.vars[2]

        return [tf.losses.absolute_difference(labels=self.vars[0], predictions=self.vars[1], weights=weights)]

    def execute_cosine_distance(self):
        weights = 1.0
        if(len(self.vars) > 2):
            weights = self.vars[2]
        r = self.op.get("reduction", tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

        return [tf.losses.cosine_distance(labels=self.vars[0], predictions=self.vars[1], weights=weights, axis=self.op["axis"], reduction=r)]

    def execute_hinge_loss(self):
        weights = 1.0
        if(len(self.vars) > 2):
            weights = self.vars[2]
        r = self.op.get("reduction", tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

        return [tf.losses.hinge_loss(labels=self.vars[0], logits=self.vars[1], weights=weights, reduction=r)]

    def execute_huber_loss(self):
        weights = 1.0
        if(len(self.vars) > 2):
            weights = self.vars[2]
        r = self.op.get("reduction", tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
        delta = self.op.get("delta", 1.0)

        return [tf.losses.huber_loss(labels=self.vars[0], predictions=self.vars[1], weights=weights, reduction=r, delta=delta)]

    def execute_log_loss(self):
        weights = 1.0
        if(len(self.vars) > 2):
            weights = self.vars[2]
        r = self.op.get("reduction", tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
        eps = self.op.get("epsilon", 1e-7)

        return [tf.losses.log_loss(labels=self.vars[0], predictions=self.vars[1], weights=weights, reduction=r, epsilon=eps)]

    def execute_sigmoid_cross_entropy(self):
        weights = 1.0
        if(len(self.vars) > 2):
            weights = self.vars[2]
        ls = self.op.get("label_smoothing",0)
        r = self.op.get("reduction", tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

        return [tf.losses.sigmoid_cross_entropy(multi_class_labels=self.vars[0], logits=self.vars[1], weights=weights, label_smoothing=ls, reduction=r)]

    def execute_softmax_cross_entropy(self):
        weights = 1.0
        if(len(self.vars) > 2):
            weights = self.vars[2]
        ls = self.op.get("label_smoothing",0)
        r = self.op.get("reduction", tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

        return [tf.losses.softmax_cross_entropy(onehot_labels=self.vars[0], logits=self.vars[1], weights=weights, label_smoothing=ls, reduction=r)]

    def execute_sparse_softmax_cross_entropy(self):
        weights = 1.0
        if(len(self.vars) > 2):
            weights = self.vars[2]
        r = self.op.get("reduction", tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

        return [tf.losses.sparse_softmax_cross_entropy(labels=self.vars[0], logits=self.vars[1], weights=weights, reduction=r)]

    def execute_l2_loss(self):
        return [tf.nn.l2_loss(self.vars[0])]

    def execute_nn_cnn1d(self):
        return [tf.nn.conv1d(value=self.vars[0], filters=self.vars[1], stride=self.op["stride"], padding=self.op["padding"], data_format=self.op["data_format"])]

    def execute_layers_cnn1d(self):
        kr = self.op.get("kernel_regularizer",None)
        br = self.op.get("bias_regularizer",None)
        ar = self.op.get("activity_regularizer",None)
        kc = self.op.get("kernel_constraint",None)
        bc = self.op.get("bias_constraint",None)
        print("kernel constraint: ", kc)
        print("bias constraint: ", bc)
        return [tf.layers.conv1d(inputs=self.vars[0], filters=self.op["filters"], kernel_size=self.op["kernel_size"], strides=self.op["strides"],
                                 padding=self.op["padding"], data_format=self.op["data_format"], dilation_rate=self.op["dilation_rate"],
                                 kernel_regularizer=kr, bias_regularizer=br, activity_regularizer=ar, kernel_constraint=kc, bias_constraint=bc)]

    def execute_max_pooling1d(self):
        return [tf.layers.max_pooling1d(inputs=self.vars[0], pool_size=self.op["pooling_size"], strides=self.op["stride"], padding=self.op["padding"], data_format=self.op["data_format"])]

    def execute_avg_pooling1d(self):
        return [tf.layers.average_pooling1d(inputs=self.vars[0], pool_size=self.op["pooling_size"], strides=self.op["stride"], padding=self.op["padding"], data_format=self.op["data_format"])]

    def execute_dense(self):
        kr = self.op.get("kernel_regularizer",None)
        br = self.op.get("bias_regularizer",None)
        return [tf.layers.dense(inputs=self.vars[0], units=self.op["units"], activation=self.op["activation"], use_bias=self.op["use_bias"], kernel_regularizer=kr, bias_regularizer=br)]

    def execute_flatten(self):
        return [tf.layers.flatten(inputs=self.vars[0])]

    def execute_nn_conv2d(self):
        return [tf.nn.conv2d(input=self.vars[0], filter=self.vars[1], strides=self.op["strides"], padding=self.op["padding"],
                             data_format=self.op["data_format"], dilations=self.op.get("dilations", [1,1,1,1]))]

    def execute_layers_conv2d(self):
        return [tf.layers.conv2d(inputs=self.vars[0], filters=self.op["filters"], kernel_size=self.op["kernel_size"], strides=self.op["strides"],
                                 padding=self.op["padding"], data_format=self.op["data_format"], dilation_rate=self.op["dilation_rate"],
                                 activation=self.op.get("activation",None), kernel_regularizer=self.op.get("kernel_regularizer",None),
                                 bias_regularizer=self.op.get("bias_regularizer",None), activity_regularizer=self.op.get("activity_regularizer",None),
                                 kernel_constraint=self.op.get("kernel_constraint",None), bias_constraint=self.op.get("bias_constraint",None))]

    def execute_layers_sepconv1d(self):
        return [tf.layers.separable_conv1d(inputs=self.vars[0], filters=self.op["filters"], kernel_size=self.op["kernel_size"], strides=self.op["strides"],
                                           padding=self.op["padding"], data_format=self.op["data_format"], dilation_rate=self.op["dilation_rate"],
                                           depth_multiplier=self.op["depth_multiplier"],
                                           activation=self.op.get("activation",None), depthwise_regularizer=self.op.get("kernel_regularizer",None),
                                           bias_regularizer=self.op.get("bias_regularizer",None), activity_regularizer=self.op.get("activity_regularizer",None),
                                           depthwise_constraint=self.op.get("kernel_constraint",None), bias_constraint=self.op.get("bias_constraint",None))]

    def execute_layers_sepconv2d(self):
        return [tf.layers.separable_conv1d(inputs=self.vars[0], filters=self.op["filters"], kernel_size=self.op["kernel_size"], strides=self.op["strides"],
                                           padding=self.op["padding"], data_format=self.op["data_format"], dilation_rate=self.op["dilation_rate"],
                                           depth_multiplier=self.op["depth_multiplier"],
                                           activation=self.op.get("activation",None), depthwise_regularizer=self.op.get("kernel_regularizer",None),
                                           bias_regularizer=self.op.get("bias_regularizer",None), activity_regularizer=self.op.get("activity_regularizer",None),
                                           depthwise_constraint=self.op.get("kernel_constraint",None), bias_constraint=self.op.get("bias_constraint",None))]

    def execute_layers_conv2d_transpose(self):
        return [tf.layers.conv2d_transpose(inputs=self.vars[0], filters=self.op["filters"], kernel_size=self.op["kernel_size"], strides=self.op["strides"],
                                 padding=self.op["padding"], data_format=self.op["data_format"],
                                 activation=self.op.get("activation",None), kernel_regularizer=self.op.get("kernel_regularizer",None),
                                 bias_regularizer=self.op.get("bias_regularizer",None), activity_regularizer=self.op.get("activity_regularizer",None),
                                 kernel_constraint=self.op.get("kernel_constraint",None), bias_constraint=self.op.get("bias_constraint",None))]


    def execute_layers_conv3d(self):
        return [tf.layers.conv3d(inputs=self.vars[0], filters=self.op["filters"], kernel_size=self.op["kernel_size"], strides=self.op["strides"],
                                 padding=self.op["padding"], data_format=self.op["data_format"], dilation_rate=self.op["dilation_rate"],
                                 activation=self.op.get("activation",None), kernel_regularizer=self.op.get("kernel_regularizer",None),
                                 bias_regularizer=self.op.get("bias_regularizer",None), activity_regularizer=self.op.get("activity_regularizer",None),
                                 kernel_constraint=self.op.get("kernel_constraint",None), bias_constraint=self.op.get("bias_constraint",None))]

    def execute_max_pooling3d(self):
        return [tf.layers.max_pooling3d(inputs=self.vars[0], pool_size=self.op["pooling_size"], strides=self.op["stride"], padding=self.op["padding"], data_format=self.op["data_format"])]

    def execute_avg_pooling3d(self):
        return [tf.layers.average_pooling3d(inputs=self.vars[0], pool_size=self.op["pooling_size"], strides=self.op["stride"], padding=self.op["padding"], data_format=self.op["data_format"])]

    def execute_batchnorm(self):
        return [tf.layers.batch_normalization(inputs=self.vars[0], axis=self.op["axis"], momentum=self.op.get("momentum",0.99), epsilon=self.op.get("epsilon",0.001),
                                              center=self.op.get("center",True), scale=self.op.get("scale",True), fused=self.op["fused"])]

    def execute_embedding_lookup(self):
        nParamArrs = len(self.vars)-1
        params = []
        for i in range(nParamArrs):
            params.append(self.vars[i])
        print("vars: ", self.vars)
        print("ids: ", self.vars[nParamArrs])
        return [tf.nn.embedding_lookup(params=params, ids=self.vars[nParamArrs], partition_strategy=self.op["partition_strategy"], max_norm=self.op["max_norm"])]

    def execute_l2_normalize(self):
        return [tf.nn.l2_normalize(x=self.vars[0], axis=self.op["axis"], epsilon=self.op["epsilon"])]

    def execute_lrn(self):
        return [tf.nn.lrn(input=self.vars[0], depth_radius=self.op["depth_radius"], bias=self.op["bias"], alpha=self.op["alpha"], beta=self.op["beta"])]

    def execute_layers_dropout(self):
        return [tf.layers.dropout(inputs=self.vars[0], rate=self.op["rate"], noise_shape=self.op.get("noise_shape",None), training=self.op["training"])]

    def execute_contrib_nn_alpha_dropout(self):
        return [tf.contrib.nn.alpha_dropout(x=self.vars[0], keep_prob=self.op["keep_prob"], noise_shape=self.op.get("noise_shape",None))]

    def execute_meshgrid(self):
        meshgrid = tf.meshgrid(self.vars, indexing=self.op["indexing"])
        return [tf.stack(meshgrid, axis=0)] #Workaround for multi-output issue

    def execute_eye(self):
        batch_shape = None
        if(len(self.vars) > 0):
            batch_shape = tf.cast(self.vars[0],dtype=tf.int32)
        return [tf.eye(num_rows=self.op["num_rows"], num_columns=self.op["num_columns"], batch_shape=batch_shape)]