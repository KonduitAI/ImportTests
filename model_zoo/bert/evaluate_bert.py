import tensorflow as tf
import tokenization
import numpy as np


class Tokenizer:
    def tokenize(str):

        tokenizer = tokenization.FullTokenizer(vocab_file=vocabFile, do_lower_case=True)
        bert_tokens = []
        bert_tokens.append("[CLS]")
        out = tokenizer.tokenize(str)
        print(out)
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

if __name__ == '__main__':

    '''
    First: Parse with vocab
    
    As far as format, BERT has 3 inputs:
    (a) sequence IDs - INT64, shape [mb, maxSeqLength], zero padded 
    (b) Mask - INT64, value 1 where data is present, 0 otherwise. Zero padded
    (c) Segment IDs: for sentence classification (not sentence matching task) this should be all 0s
    
    TF graph has the useful input names "Placeholder", "Placeholder_1", "Placeholder_2"
    '''

    vocabFile = "/TF_Graphs/BERT_multi_cased_L-12_H-768_A-12/vocab.txt"
    str = "To be, or not to be, that is the question:"
    str2 = "Whether 'tis nobler in the mind to suffer"
    idxs = Tokenizer.tokenize(str)
    idxs2 = Tokenizer.tokenize(str2)

    minLength = min(len(idxs), len(idxs2))
    maxLength = max(len(idxs), len(idxs2))

    print(idxs)
    print(idxs2)
    print(minLength)
    print(maxLength)

    fArr = np.zeros([2,15], dtype=np.int64)
    for i in range(len(idxs)):
        fArr[0,i] = idxs[i]
    for i in range(len(idxs2)):
        fArr[1,i] = idxs2[i]
    mArr = np.ones([2,15], dtype=np.int64)
    mArr[0][14] = 0

    segmentArr = np.zeros([2,15], dtype=np.int64)

    print(fArr)
    print(mArr)
    print(segmentArr)

    #Load network
    with tf.gfile.GFile("/TF_Graphs/BERT_multi_cased_L-12_H-768_A-12/BERT_multi_cased_L-12_H-768_A-12_frozen.pb", "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    print(graph)

    for op in graph.get_operations():
        print(op.name)

    feed_dict = {
        "Placeholder:0": fArr,
        "Placeholder_1:0": mArr,
        "Placeholder_2:0": segmentArr,
    }
    print(feed_dict)

    outputNames = ["bert/pooler/dense/Tanh"]

    with tf.Session(graph=graph) as sess:
        outputs = sess.run(
            outputNames,
            feed_dict=feed_dict)
        print("OUTPUTS: ", outputs)