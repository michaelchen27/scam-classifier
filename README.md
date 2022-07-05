# Scam Classifier Model Training using Federated Learning Method
Machine learning model training in a federated fashion tasked to do Binary Classification between scam and non-scam message.

## Description
This repository contains training script to train a binary classifier model to classify whether a given message is a scam or not a scam. The model is trained with two kinds of Federated Learning method: The common Federated Learning method that uses an aggregation algorithm and a simple-daisy chaining method that doesn't apply the aggregation algorithm.

## Getting Started
### Dependencies

* PySyft v0.2.9
* PyTorch v1.4.0
* Python v3.7.1
* Numpy
* Scikit-learn
* Matplotlib
* Pandas
* NLTK
* Regex

### Installing Core Dependencies
* Installing PySyft + PyTorch
```
pip install syft==0.2.9 -f https://download.pytorch.org/whl/torch_stable.html
```

## Project Files
* `Federated Learning with Pysyft and Pytorch.ipynb`
The main notebook that contains both the training script to train the classifier model.
* `handcrafted_GRU.py`
The main model with pre-defined network layers imported by the main notebook. The network layers defined are: Dropout Layer, Embedding Layer, GRU Cell, a single Fully-connected Layer, and a layer with Sigmoid activation before output layer.
* `preprocess.py`
Dataset pre-process script to clean and uniformize the data before feeding it into the model. The process includes these steps: Text lowering, symbol removal, Indonesian stopwords removal, word tokenization, and padding or truncation.
* `inputs.npy`
Input data that contains the tokenized text message in a form of Numpy Binary File produced by the pre-processing script.
* `labels.npy`
Label data that contains the labels or class for each text message in a form of Numpy Binary File produced by the pre-processing script.
* `test_inputs.npy`
Pre-processed input data to test the model.
* `test_labels.npy`
Pre-processed label data to test and calculate the performance of the model.
* `train_sms_1240.csv`
Train dataset to train and evaluate the model.
* `test_sms_540.csv`
Test dataset to test and calculate the model performance.

Train and Test Dataset Format:
```
"[Text]", [Label]

Example:
"This is a scam message, pls clickz  bit.ly/81h3s thenkyou!", 1
"Hello, this is a normal conversation message", 0
"LIMITED SALE! Buy iPhone WITH 1-YEAR DATA QUOTA! Visit the nearest iBox!", 0
```

* `local_state_dict_model.pt`
Previous model trained with the common Federated Learning method that includes the aggregation algorithm.
* `state_dict_model.pt`
Previous model trained with the simple-daisy chaining Federated Learning method that doesn't involve an aggregation algorithm.

## Help

Some package library installs their required dependency automatically in the installation process. This process might overwrite the currently installed package library version with the same name. It's recommended to first install both PyTorch and PySyft with the exact same version mentioned above.

## Authors
Contact:
Michael C.
Twitter @michen27

## Acknowledgments

* [Medium: Private-AI](https://towardsdatascience.com/private-ai-federated-learning-with-pysyft-and-pytorch-954a9e4a4d4e)
* [Federated Learning from Small Datasets](https://arxiv.org/pdf/2110.03469.pdf)
* [Opacus+PySyft - A short demo](https://web.archive.org/web/20210623140627/https://blog.openmined.org/pysyft-opacus-federated-learning-with-differential-privacy/)
