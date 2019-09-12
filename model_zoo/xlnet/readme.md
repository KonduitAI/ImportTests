# How to reproduce the XLNET frozen graph

Clone the XLNet repo: https://github.com/zihangdai/xlnet.git to somewhere inside this project.

### Edit `run_classifier.py`:
At line 718, where the `train_input_fn` is created, add:
```python
with tf.Session().as_default():
    itr = tf.data.make_one_shot_iterator(train_input_fn({}))
    input_1 = np.zeros((8, 128), 'int32')
    while np.count_nonzero(np.sum(input_1, axis=1) < 10) > 5:
        data = itr.get_next()
        input = data['input_ids'].eval()
        input_1 = data['input_mask'].eval()
        input_2 = data['segment_ids'].eval()

    np.save("C:\\Skymind\\TFOpTests\\model_zoo\\util\\data\\xlnet\\input.npy", input)
    np.save("C:\\Skymind\\TFOpTests\\model_zoo\\util\\data\\xlnet\\input_1.npy", input_1)
    np.save("C:\\Skymind\\TFOpTests\\model_zoo\\util\\data\\xlnet\\input_2.npy", input_2)
```

With appropriate save files.

Also, at line ~765, change `global_step = int(cur_filename.split("-")[-1])` to `global_step = 0`.

### Edit `function_builder.py`:
Import `from model_zoo.xlnet.freeze_xlnet import freeze_xlnet` (from TFOpTests).

Add to `get_classification_loss`, right before the return (line ~173):
```python
freeze_xlnet([logits.name.replace(':0', '')], graph=tf.get_default_graph())
```

`freeze_xlnet` will need to have its files updated.

I suggest putting a breakpoint or a `quit()` after the `freeze_xlnet` model, as it saves the model and you don't need to do anything more.

To save the classifier model, run `run_classifier.py` with the arguments 
```--do_train=True --do_eval=True --task_name=imdb --data_dir=C:\Temp\TF_Graphs\xlnet_cased_L-24_H-1024_A-16\aclImdb\ --output_dir=proc_data/imdb --model_dir=model --uncased=False --spiece_model_file=C:\Temp\TF_Graphs\xlnet_cased_L-24_H-1024_A-16\spiece.model --model_config_path=C:\Temp\TF_Graphs\xlnet_cased_L-24_H-1024_A-16\xlnet_config.json --init_checkpoint=C:\Temp\TF_Graphs\xlnet_cased_L-24_H-1024_A-16\xlnet_model.ckpt --max_seq_length=128 --eval_batch_size=8 --num_hosts=1 --num_core_per_host=1 --learning_rate=2e-5 -- train_steps=40 -- warmup_steps=10 --save_steps=50 --iterations=50```

Directories should be adjusted, and you will have to preprocess data.  Treat this just like an XLNet training run.