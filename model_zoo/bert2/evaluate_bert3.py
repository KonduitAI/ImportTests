import tensorflow as tf
import tokenization
import numpy as np
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
import tensorflow_hub as hub


BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
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

        MAX_SEQ_LENGTH=128
        label_list = [0, 1]
        do_lower_case = True
        vocab_file = "/TF_Graphs/BERT_multi_cased_L-12_H-768_A-12/vocab.txt"
        tokenizer = bert.tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

        in_sentences = ["To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer"]
        labels = ["Negative", "Positive"]
        input_examples = [run_classifier.InputExample(guid="", text_a = x, text_b = None, label = 0) for x in in_sentences] # here, "" is just a dummy label
        features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)

        print("input_examples: ", input_examples)
        print("input_features: ", features)

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        print("input_ids: ", input_ids)
        print("input_mask: ", input_mask)
        print("segment_ids: ", segment_ids)
        print("label_ids: ", label_ids)

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

    def create_model(is_predicting, input_ids, input_mask, segment_ids, labels,
                     num_labels):
        """Creates a classification model."""

        bert_module = hub.Module(
            BERT_MODEL_HUB,
            trainable=True)
        bert_inputs = dict(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids)
        bert_outputs = bert_module(
            inputs=bert_inputs,
            signature="tokens",
            as_dict=True)

        # Use "pooled_output" for classification tasks on an entire sentence.
        # Use "sequence_outputs" for token-level output.
        output_layer = bert_outputs["pooled_output"]

        hidden_size = output_layer.shape[-1].value

        # Create our own layer to tune for politeness data.
        output_weights = tf.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer())

        with tf.variable_scope("loss"):

            # Dropout helps prevent overfitting
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            # Convert labels into one-hot encoding
            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

            predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
            # If we're predicting, we want predicted labels and the probabiltiies.
            if is_predicting:
                return (predicted_labels, log_probs)

            # If we're train/eval, compute loss between predicted and actual label
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)
            return (loss, predicted_labels, log_probs)

if __name__ == '__main__':
    graph = EvalBert.evaluate("/TF_Graphs/BERT_multi_cased_L-12_H-768_A-12/bert_model.ckpt")