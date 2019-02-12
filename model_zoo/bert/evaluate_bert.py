import tensorflow as tf
import tokenization
import numpy as np


class Tokenizer:
    def tokenize(str):

        tokenizer = tokenization.FullTokenizer(vocab_file=vocabFile, do_lower_case=True)
        bert_tokens = []
        bert_tokens.append("[CLS]")
        out = tokenizer.tokenize(str)
        # print(out)
        for t in out:
            bert_tokens.append(t)
        bert_tokens.append("[SEP]")
        print(bert_tokens)

        idxs = []
        for t in bert_tokens:
            id = tokenizer.vocab[t]
            idxs.append(id)
        # print(idxs)
        return idxs

    def load_graph(checkpoint_path):
        init_all_op = tf.initialize_all_variables()
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as sess:
                saver = tf.train.import_meta_graph(checkpoint_path + '.meta')
                saver.restore(sess, checkpoint_path)
                print("Restored structure...")
                saver.restore(sess, checkpoint_path)
                print("Restored params...")

                # for op in graph2.get_operations():
                #     print(op.name)
                return graph

if __name__ == '__main__':

    '''
    First: Parse with vocab
    
    As far as format, BERT has 3 inputs:
    (a) sequence IDs - INT64? INT32?, shape [mb, maxSeqLength], zero padded 
    (b) Mask - INT64, value 1 where data is present, 0 otherwise. Zero padded
    (c) Segment IDs: for sentence classification (not sentence matching task) this should be all 0s
    
    TF graph has the useful input names "Placeholder", "Placeholder_1", "Placeholder_2"
    
    NOTE: BERT seems to expect exactly 128 length inputs, always... no variable length here, just truncate and mask
    '''

    vocabFile = "/TF_Graphs/BERT_multi_cased_L-12_H-768_A-12/vocab.txt"
    str = "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer"
    idxs = Tokenizer.tokenize(str)

    # fArr = np.zeros([1,128], dtype=np.int64)
    fArr = np.zeros([1,128], dtype=np.int32)
    for i in range(len(idxs)):
        fArr[0,i] = idxs[i]
    # mArr = np.zeros([1,128], dtype=np.int64)
    mArr = np.zeros([1,128], dtype=np.int32)
    for i in range(128):
        if(i < len(idxs)):
            mArr[0,i] = 1

    # segmentArr = np.zeros([1,128], dtype=np.int64)
    segmentArr = np.zeros([1,128], dtype=np.int32)

    # segmentArr[0,0] = 1
    # segmentArr[0,1] = 1
    # segmentArr[0,2] = 1
    # segmentArr[0,3] = 1

    #Load network from frozen model:
    with tf.gfile.GFile("/TF_Graphs/BERT_multi_cased_L-12_H-768_A-12/BERT_multi_cased_L-12_H-768_A-12_frozen.pb", "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    #Load network from original model:
    # graph = Tokenizer.load_graph("/TF_Graphs/BERT_multi_cased_L-12_H-768_A-12/bert_model.ckpt")

    # print(graph)
    # tf.global_variables_initializer()

    # for op in graph.get_operations():
    #     print(op.name)

    # TODO - not 100% sure on the order here... the BERT code doesn't contain the text "Placeholder" anywhere...
    feed_dict = {
        "Placeholder:0": fArr,
        "Placeholder_1:0": mArr,
        "Placeholder_2:0": segmentArr,
    }
    print(feed_dict)

    # outputNames = ["bert/encoder/Reshape_13", "bert/pooler/dense/Tanh", "bert/encoder/layer_0/output/LayerNorm/batchnorm/add_1", "bert/embeddings/add"]
    outputNames = ["bert/pooler/dense/Tanh"]

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer()) #For non-frozen model: "failed to get matching files" :/ But without it: "Attempting to use uninitialized value"
        outputs = sess.run(outputNames,feed_dict=feed_dict)
        print("OUTPUTS: ", outputs)