# How to reproduce the GPT-2 frozen graph

Clone the GPT-2 repo https://github.com/openai/gpt-2.git somewhere it can be imported.

You will need to be able to import the `model.py`, `sample.py` and `encoder.py` files.

You will need to adjust paths in `freeze_model.py` and make sure the imports work (PyCharm will likely flag them as errors even if they will work), then run it.  

It is ran just like `generate_unconditional_samples.py`, and so will need the same preparation (e.g. running `download_model.py 117M`).