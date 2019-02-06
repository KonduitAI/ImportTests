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

    def execute_log_determinant(self):
        #Attempting to ensure the input sub-matrices are hermitian positive definite matrix... this doesn't guarantee it??
        inArr = self.vars[0]
        if(len(self.op["varShapes"][0]) == 2):
            inArr = inArr + tf.eye(num_rows=self.op["varShapes"][0][0], num_columns=self.op["varShapes"][0][1])
        elif(len(self.op["varShapes"][0]) == 3):
            inArr = inArr + tf.eye(num_rows=self.op["varShapes"][0][1], num_columns=self.op["varShapes"][0][2], batch_shape=[self.op["varShapes"][0][0]])
        elif(len(self.op["varShapes"][0]) == 4):
            inArr = inArr + tf.eye(num_rows=self.op["varShapes"][0][2], num_columns=self.op["varShapes"][0][3], batch_shape=[self.op["varShapes"][0][0], self.op["varShapes"][0][1]])
        else:
            raise ValueError("Only rank 2-4 implemented")

        return [tf.linalg.logdet(inArr)]

    def execute_slog_determinant(self):
        #Attempting to ensure the input sub-matrices are hermitian positive definite matrix... this doesn't guarantee it??
        inArr = self.vars[0]
        if(len(self.op["varShapes"][0]) == 2):
            inArr = inArr + tf.eye(num_rows=self.op["varShapes"][0][0], num_columns=self.op["varShapes"][0][1])
        elif(len(self.op["varShapes"][0]) == 3):
            inArr = inArr + tf.eye(num_rows=self.op["varShapes"][0][1], num_columns=self.op["varShapes"][0][2], batch_shape=[self.op["varShapes"][0][0]])
        elif(len(self.op["varShapes"][0]) == 4):
            inArr = inArr + tf.eye(num_rows=self.op["varShapes"][0][2], num_columns=self.op["varShapes"][0][3], batch_shape=[self.op["varShapes"][0][0], self.op["varShapes"][0][1]])
        else:
            raise ValueError("Only rank 2-4 implemented")

        return tf.linalg.slogdet(inArr)


    def execute_sequence_mask(self):
        maxLen = None
        if(len(self.vars) > 1):
            maxLen = self.vars[1]
        return [tf.sequence_mask(lengths=self.vars[0], maxlen=maxLen)]

    def execute_rint(self):
        return [tf.rint(self.vars[0])]

    def execute_histogram_fixed_width(self):
        return [tf.histogram_fixed_width(values=self.vars[0], value_range=self.vars[1], nbins=self.op["nbins"])]

    def execute_bincount(self):
        w = None
        if(len(self.vars) > 1):
            w = self.vars[1]
        return [tf.bincount(arr=self.vars[0], weights=w, minlength=self.op["minlength"], maxlength=self.op["maxlength"])]

    def execute_scatter_nd(self):
        return [tf.scatter_nd(indices=self.vars[0], updates=self.vars[1], shape=self.op["shape"])]

    def execute_scatter_nd_add(self):
        # Create an intermediate variable - otherwise the scatter op will modify the variable content in-place
        # and hence we'll save the input post-modification, rather than pre-modification
        intermediate = tf.Variable(tf.zeros(self.op["varShapes"][0]), dtype=tf.float32)
        intermediate = tf.assign(intermediate, self.vars[0])
        return [tf.scatter_nd_add(ref=intermediate, indices=self.vars[1], updates=self.vars[2])]

    def execute_scatter_nd_sub(self):
        intermediate = tf.Variable(tf.zeros(self.op["varShapes"][0]), dtype=tf.float32)
        intermediate = tf.assign(intermediate, self.vars[0])
        return [tf.scatter_nd_sub(ref=intermediate, indices=self.vars[1], updates=self.vars[2])]

    def execute_scatter_nd_update(self):
        intermediate = tf.Variable(tf.zeros(self.op["varShapes"][0]), dtype=tf.float32)
        intermediate = tf.assign(intermediate, self.vars[0])
        return [tf.scatter_nd_update(ref=intermediate, indices=self.vars[1], updates=self.vars[2])]

    def execute_sufficient_statistics(self):
        temp = tf.add(self.vars[0], 1.0)
        return tf.nn.sufficient_statistics(x=self.vars[0], axes=self.op["axes"], shift=self.op["shift"], keep_dims=self.op["keep_dims"])

    def execute_split(self):
        num_or_size_splits=self.op.get("num_or_size_split", None)
        return tf.split(value=self.vars[0], num_or_size_splits=num_or_size_splits, axis=self.op["axis"])

    def execute_reduce_logsumexp(self):
        return [tf.reduce_logsumexp(input_tensor=self.vars[0], axis=self.op["axis"], keep_dims=self.op["keep_dims"])]

    def execute_nth_element(self):
        return [tf.contrib.nn.nth_element(input=self.vars[0], n=self.vars[1], reverse=self.op["reverse"])]

    def execute_reduce_any(self):
        return [tf.reduce_any(input_tensor=self.vars[0], axis=self.op["axis"], keep_dims=self.op["keep_dims"])]

    def execute_reduce_all(self):
        return [tf.reduce_all(input_tensor=self.vars[0], axis=self.op["axis"], keep_dims=self.op["keep_dims"])]

    def execute_boolean_mask(self):
        return [tf.boolean_mask(tensor=self.vars[0], mask=self.vars[1])]

    def execute_where(self):
        c = self.vars[0]
        x = None
        y = None
        if(len(self.vars) > 1):
            x = self.vars[1]
            y = self.vars[2]
        else:
            tf.Variable(tf.add(self.vars[0], 0.0))
        # print("x: ",x)
        # print("y: ",y)
        # print("cond: ",c)
        return [tf.where(condition=c, x=x, y=y)]

    def execute_broadcast_dynamic_shape(self):
        return [tf.broadcast_dynamic_shape(self.vars[0], self.vars[1])]

    def execute_broadcast_to(self):
        return [tf.broadcast_to(input=self.vars[0], shape=self.vars[1])]

    def execute_unsorted_segment_max(self):
        return [tf.unsorted_segment_max(data=self.vars[0], segment_ids=self.vars[1], num_segments=self.op["num_segments"])]

    def execute_unsorted_segment_min(self):
        return [tf.unsorted_segment_min(data=self.vars[0], segment_ids=self.vars[1], num_segments=self.op["num_segments"])]

    def execute_unsorted_segment_mean(self):
        return [tf.unsorted_segment_mean(data=self.vars[0], segment_ids=self.vars[1], num_segments=self.op["num_segments"])]

    def execute_unsorted_segment_prod(self):
        return [tf.unsorted_segment_prod(data=self.vars[0], segment_ids=self.vars[1], num_segments=self.op["num_segments"])]

    def execute_unsorted_segment_sqrt_n(self):
        return [tf.unsorted_segment_sqrt_n(data=self.vars[0], segment_ids=self.vars[1], num_segments=self.op["num_segments"])]

    def execute_unsorted_segment_sum(self):
        return [tf.unsorted_segment_sum(data=self.vars[0], segment_ids=self.vars[1], num_segments=self.op["num_segments"])]

    def execute_truncatemod(self):
        return [tf.truncatemod(x=self.vars[0], y=self.vars[1])]

    def execute_tensordot(self):
        return [tf.tensordot(a=self.vars[0], b=self.vars[1], axes=self.op["axes"])]

    def execute_assert_equal(self):
        with tf.control_dependencies([tf.assert_equal(x=self.vars[0], y=self.vars[1])]):
            out = tf.add(self.vars[0], self.vars[1])
        return [out]

    def execute_assert_greater(self):
        with tf.control_dependencies([tf.assert_greater(x=self.vars[0], y=self.vars[1])]):
            out = tf.add(self.vars[0], self.vars[1])
        return [out]

    def execute_assert_greater_equal(self):
        with tf.control_dependencies([tf.assert_greater_equal(x=self.vars[0], y=self.vars[1])]):
            out = tf.add(self.vars[0], self.vars[1])
        return [out]

    def execute_assert_less(self):
        with tf.control_dependencies([tf.assert_less(x=self.vars[0], y=self.vars[1])]):
            out = tf.add(self.vars[0], self.vars[1])
        return [out]

    def execute_assert_less_equal(self):
        with tf.control_dependencies([tf.assert_less_equal(x=self.vars[0], y=self.vars[1])]):
            out = tf.add(self.vars[0], self.vars[1])
        return [out]

    def execute_assert_none_equal(self):
        with tf.control_dependencies([tf.assert_none_equal(x=self.vars[0], y=self.vars[1])]):
            out = tf.add(self.vars[0], self.vars[1])
        return [out]

    def execute_assert_integer(self):
        with tf.control_dependencies([tf.assert_integer(x=self.vars[0])]):
            out = tf.add(self.vars[0], 1)
        return [out]

    def execute_assert_negative(self):
        with tf.control_dependencies([tf.assert_negative(x=self.vars[0])]):
            out = tf.add(self.vars[0], 1)
        return [out]

    def execute_assert_positive(self):
        with tf.control_dependencies([tf.assert_positive(x=self.vars[0])]):
            out = tf.add(self.vars[0], 1)
        return [out]

    def execute_assert_rank(self):
        with tf.control_dependencies([tf.assert_rank(x=self.vars[0], rank=self.vars[1])]):
            out = tf.add(self.vars[0], tf.cast(self.vars[1], self.vars[0].dtype))
        return [out]

    def execute_assert_rank_at_least(self):
        with tf.control_dependencies([tf.assert_rank_at_least(x=self.vars[0], rank=self.vars[1])]):
            out = tf.add(self.vars[0], tf.cast(self.vars[1], self.vars[0].dtype))
        return [out]

    def execute_assert_type(self):
        with tf.control_dependencies([tf.assert_type(tensor=self.vars[0], tf_type=self.op["tf_type"])]):
            out = tf.add(self.vars[0], 1)
        return [out]

    def execute_cond(self):
        def ifTrue():
            return tf.lin_space(start=1.0, stop=5.0, num=5)
        def ifFalse():
            return tf.ones(shape=[5], dtype=tf.float32)
        return [tf.cond(self.vars[0], ifTrue, ifFalse)]

    def execute_case(self):
        input = self.vars[0]
        a = (input <= 1, lambda: input * 1)
        b = (input <= 2, lambda: input * 2)
        c = (input <= 3, lambda: input * 3)
        default = lambda: input * 4
        pairs = [a,b,c]
        return [tf.case(pairs, default)]

    def execute_while1(self):
        # Simple counter loop, there condition is less than self.vars[1]
        def condition(i, j):
            return i < j
        def body(i, j):
            return i+1, j
        loop = tf.while_loop(condition, body, (0.0, self.vars[0]))
        return loop

    def execute_while2(self):
        # Loop: keep dividing self.vars[1] by 2 until sum(self.vars[1]) < sum(self.vars[0])
        def condition(x, y):
            return tf.reduce_sum(y) < tf.reduce_sum(x)
        def body(x, y):
            return x, y/2
        loop = tf.while_loop(condition, body, (self.vars[0], self.vars[1]))
        return loop

    def execute_sum_dynamic_axis(self):
        if(self.op["axistype"] == "argmin"):
            axis = tf.math.argmin(tf.shape(self.vars[0]))
        else:
            axis = tf.math.argmax(tf.shape(self.vars[0]))
        return [tf.reduce_sum(self.vars[0], axis=axis, keepdims=self.op["keepdims"])]

    def execute_tensorarray_getset(self):
        infershape = True
        if("infer_shape" in self.op):
            infershape = self.op["infer_shape"]
        ta = tf.TensorArray(dtype=self.op["dtype"], size=self.op["size"], dynamic_size=self.op["dynamic_size"], tensor_array_name=self.op["tensor_array_name"], element_shape=self.op["element_shape"], infer_shape=infershape)
        n = len(self.vars)
        for i in range(n):
            #Note: on each write, need to use the new/returned TensorArray for all subsequent ops
            ta = ta.write(i, self.vars[i])

        out = []
        for i in range(n):
            out.append(ta.read(i))
        return out

    def execute_tensorarray_size(self):
        infershape = True
        if("infer_shape" in self.op):
            infershape = self.op["infer_shape"]
        ta = tf.TensorArray(dtype=self.op["dtype"], size=self.op["size"], dynamic_size=self.op["dynamic_size"], tensor_array_name=self.op["tensor_array_name"], element_shape=self.op["element_shape"], infer_shape=infershape)
        n = len(self.vars)
        for i in range(n):
            #Note: on each write, need to use the new/returned TensorArray for all subsequent ops
            ta = ta.write(i, self.vars[i])

        return [ta.size()]

    def execute_tensorarray_concat(self):
        infershape = True
        if("infer_shape" in self.op):
            infershape = self.op["infer_shape"]
        ta = tf.TensorArray(dtype=self.op["dtype"], size=self.op["size"], dynamic_size=self.op["dynamic_size"], tensor_array_name=self.op["tensor_array_name"], element_shape=self.op["element_shape"], infer_shape=infershape)
        n = len(self.vars)
        for i in range(n):
            #Note: on each write, need to use the new/returned TensorArray for all subsequent ops
            ta = ta.write(i, self.vars[i])

        return [ta.concat()]

    def execute_tensorarray_stack(self):
        infershape = True
        if("infer_shape" in self.op):
            infershape = self.op["infer_shape"]
        ta = tf.TensorArray(dtype=self.op["dtype"], size=self.op["size"], dynamic_size=self.op["dynamic_size"], tensor_array_name=self.op["tensor_array_name"], element_shape=self.op["element_shape"], infer_shape=infershape)
        n = len(self.vars)
        for i in range(n):
            #Note: on each write, need to use the new/returned TensorArray for all subsequent ops
            ta = ta.write(i, self.vars[i])

        return [ta.stack()]

    def execute_tensorarray_unstack(self):
        #Unstack: create empty tensor array, stack the test array inputs, then unstack them to the TensorArray
        # (then pull them out for testing...)

        infershape = True
        if("infer_shape" in self.op):
            infershape = self.op["infer_shape"]
        ta = tf.TensorArray(dtype=self.op["dtype"], size=self.op["size"], dynamic_size=self.op["dynamic_size"], tensor_array_name=self.op["tensor_array_name"], element_shape=self.op["element_shape"], infer_shape=infershape)
        n = len(self.vars)

        stack = tf.stack(self.vars, axis=0)

        ta = ta.unstack(stack)  #Stack to increase rank by 1 before TensorArray unstack

        n = len(self.vars)
        out = []
        for i in range(n):
            #Note: on each write, need to use the new/returned TensorArray for all subsequent ops
            out.append(ta.read(i))

        return out

    def execute_tensorarray_identity(self):
        infershape = True
        if("infer_shape" in self.op):
            infershape = self.op["infer_shape"]
        ta = tf.TensorArray(dtype=self.op["dtype"], size=self.op["size"], dynamic_size=self.op["dynamic_size"], tensor_array_name=self.op["tensor_array_name"], element_shape=self.op["element_shape"], infer_shape=infershape)
        n = len(self.vars)
        for i in range(n):
            #Note: on each write, need to use the new/returned TensorArray for all subsequent ops
            ta = ta.write(i, self.vars[i])

        ta2 = ta.identity()
        out = []
        for i in range(n):
            out.append(ta2.read(i))
        return out

    def execute_tensorarray_split(self):
        infershape = True
        if("infer_shape" in self.op):
            infershape = self.op["infer_shape"]
        ta = tf.TensorArray(dtype=self.op["dtype"], size=self.op["size"], dynamic_size=self.op["dynamic_size"], tensor_array_name=self.op["tensor_array_name"], element_shape=self.op["element_shape"], infer_shape=infershape)

        ta = ta.split(value=self.vars[0], lengths=self.vars[1])

        n = self.op["varShapes"][1][0]
        out = []
        for i in range(n):
            out.append(ta.read(i))
        return out

    def execute_tensorarray_close(self):
        infershape = True
        if("infer_shape" in self.op):
            infershape = self.op["infer_shape"]
        ta = tf.TensorArray(dtype=self.op["dtype"], size=self.op["size"], dynamic_size=self.op["dynamic_size"], tensor_array_name=self.op["tensor_array_name"], element_shape=self.op["element_shape"], infer_shape=infershape)
        n = len(self.vars)
        for i in range(n):
            #Note: on each write, need to use the new/returned TensorArray for all subsequent ops
            ta = ta.write(i, self.vars[i])

        out = []
        for i in range(n):
            out.append(ta.read(i))

        ta = ta.close()     #Needs to be consumed...
        with tf.control_dependencies([ta]):
            out.append(tf.Variable(tf.ones(shape=[2,2], dtype=tf.float32)))
        return out

    def execute_extractImagePatches(self):
        out = [tf.image.extract_image_patches(images=self.vars[0], ksizes=self.op["ksizes"], strides=self.op["strides"], rates=self.op["rates"], padding=self.op["padding"])]
        return out

    def execute_stopGradient(self):
        temp = tf.tanh(self.vars[0])
        out = [tf.stop_gradient(temp)]
        return out

    def execute_lstmcell(self):
        lstm = tf.nn.rnn_cell.LSTMCell(num_units=self.op["num_units"], use_peepholes=self.op["use_peepholes"], cell_clip=self.op["cell_clip"],
                                       proj_clip=self.op["proj_clip"], forget_bias=self.op["forget_bias"], activation=self.op["activation"])

        initState = None
        if(len(self.vars) > 1):
            initState = [self.vars[1], self.vars[2]]
            if(self.op["static"] == False):
                initState = tf.nn.rnn_cell.LSTMStateTuple(initState[0], initState[1])

        if(self.op["static"] == True):
            x = tf.unstack(self.vars[0], num=self.op["timeSteps"], axis=1)
            outputs, states = tf.nn.static_rnn(lstm, inputs=x, initial_state=initState, dtype=self.op["dtype"])
        else:
            outputs, states = tf.nn.dynamic_rnn(lstm, inputs=self.vars[0], initial_state=initState, dtype=self.op["dtype"], time_major=self.op["time_major"])

        concatOutputs = tf.concat(outputs, axis=0)
        concatStates = tf.concat(states, axis=0)
        return [concatOutputs, concatStates]
