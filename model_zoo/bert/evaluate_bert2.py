import tensorflow as tf
import tokenization
import numpy as np
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
import tensorflow_hub as hub


class EvalBert:
    def tokenize(str):
        vocabFile = "/TF_Graphs/BERT_multi_cased_L-12_H-768_A-12/vocab.txt"
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

    def evaluate(checkpoint_path):
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
                '''

                MAX_SEQ_LENGTH=128
                label_list = [0, 1]
                do_lower_case = True
                vocab_file = "/TF_Graphs/BERT_multi_cased_L-12_H-768_A-12/vocab.txt"
                tokenizer = bert.tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

                in_sentences = ["To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer"]
                labels = ["Negative", "Positive"]
                input_examples = [run_classifier.InputExample(guid="", text_a = x, text_b = None, label = 0) for x in in_sentences] # here, "" is just a dummy label
                input_features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)

                print("input_examples: ", input_examples)
                print("input_features: ", input_features)
                # print("Features: ", features)

                # predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)
                # predictions = estimator.predict(predict_input_fn)
                # out = [(sentence, prediction['probabilities'], labels[prediction['labels']]) for sentence, prediction in zip(in_sentences, predictions)]
                # print("OUTPUT: ", out)

                #Load network from original model:
                # graph = Tokenizer.load_graph("/TF_Graphs/BERT_multi_cased_L-12_H-768_A-12/bert_model.ckpt")

                # print(graph)

                # for op in graph.get_operations():
                #     print(op.name)

                # TODO - not 100% sure on the order here... the BERT code doesn't contain the text "Placeholder" anywhere to help distinguish
                # All 6 possible orders give exactly the same zeros output...
                feed_dict = {
                    "Placeholder:0": fArr,
                    "Placeholder_1:0": segmentArr,
                    "Placeholder_2:0": mArr,
                }
                print(feed_dict)

                outputNames = ["bert/encoder/Reshape_13", "bert/pooler/dense/Tanh", "bert/encoder/layer_0/output/LayerNorm/batchnorm/add_1", "bert/embeddings/add"]
                # outputNames = ["bert/pooler/dense/Tanh"]

                outputs = sess.run(outputNames,feed_dict=feed_dict)
                print("OUTPUTS: ", outputs)

if __name__ == '__main__':
    graph = EvalBert.evaluate("/TF_Graphs/BERT_multi_cased_L-12_H-768_A-12/bert_model.ckpt")