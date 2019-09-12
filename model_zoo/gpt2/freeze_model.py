import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder

from model_zoo.util.freeze_keras import freeze_session


def freeze_model():
    nsamples=1,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,

    model_name='117M'
    models_dir = os.path.expanduser(os.path.expandvars('C:\\Temp\\TF_Graphs\\gpt-2\\models\\'))
    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session() as sess:
        np.random.seed(12345)
        tf.set_random_seed(12345)

        output = sample.sample_sequence(
            hparams=hparams, length=length,
            start_token=enc.encoder['<|endoftext|>'],
            batch_size=batch_size,
            temperature=temperature, top_k=top_k
        )[:, 1:]

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        generated = 0
        while nsamples == 0 or generated < nsamples:
            out = sess.run(output)
            for i in range(batch_size):
                generated += batch_size
                text = enc.decode(out[i])
                print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                print(text)

        frozen_graph = freeze_session(sess, output_names=["strided_slice"])

        base_dir = "C:\\Temp\\TF_Graphs\\gpt-2\models\\117M"

        frozen_file = "tf_model.pb"

        tf.train.write_graph(sess.graph, base_dir,
                             frozen_file + ".txt",
                             as_text=True)
        tf.train.write_graph(frozen_graph, base_dir, frozen_file, as_text=False)

        print("Output: " + output.name)

if __name__ == '__main__':
    fire.Fire(freeze_model)

