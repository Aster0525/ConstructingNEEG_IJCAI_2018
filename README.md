# Script Event Prediction with Commonsense Event Representation

This repository contains code for the script event predition task in the EMNLP 2019 paper: *Event Representation Learning Enhanced with External Commonsense Knowledge*.

This work is based on the IJCAI 2018 paper: *Constructing Narrative Event Evolutionary Graph for Script Event Prediction*. We substitute the original event representation model in this work with our model.

To run this code, you need to download files described [here](https://github.com/eecrazy/ConstructingNEEG_IJCAI_2018/blob/master/README.md) first.

Then, run the following scripts to construct the dataset for training our event representation model:

- `preproc/chains_to_dataset.py`

- `preproc/deepwalk_to_glove_format.py`

You can download the preprocessed dataset here and the converted embeddings here.

Pretrain the event representation model on the constructed dataset with the converted embeddings. (See [this repository](https://github.com/MagiaSN/commonsense_event_repr_emnlp19))

Finally, run `code/main_with_args.py` to load the pretraiend event representation model and train the SGNN model.

Our best model (including SGNN) can be downloaded here.
