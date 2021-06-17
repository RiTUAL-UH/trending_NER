# Mitigating Temporal-Drift: A Simple Approach to Keep NER Models Crisp
<p align="right"><i>Authors: Shuguang Chen, Leonardo Neves and Thamar Solorio</i></p> 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

This repository contains the implementations of the system described in the paper ["Mitigating Temporal-Drift: A Simple Approach to Keep NER Models Crisp"](https://www.aclweb.org/anthology/2021.socialnlp-1.14.pdf) on the [9th International Workshop on Natural Language Processing for Social Media](https://sites.google.com/site/socialnlp2021/) at the [NAACL](https://2021.naacl.org) conference.

## Repository Structure
```
trending_NER
├── exp_bert
│   └── src
│       ├── commons
│       │   └── utilities.py
│       ├── data # implementation of dataset class
│       ├── main.py # entire pipeline of our system
│       └── modeling
│           ├── layers.py # implementation of neural layers
│           └── nets.py # implementation of neural networks
├── exp_ssl
│   └── src
│       ├── commons
│       │   ├── globals.py
│       │   └── utilities.py
│       ├── data # implementation of dataset class
│       ├── main.py # entire pipeline of our system
│       └── modeling 
│           ├── experiment.py # entire pipeline of experiments
│           ├── nets
│           │   ├── embedding.py # implementation of embedding layers
│           │   ├── layers.py # implementation of neural layers
│           │   └── model.py # implementation of neural networks
│           └── train.py # functions to build, train, and predict with a neural network
└── notebooks
    └── 1-exploring-trending-detection.ipynb # notebook to select data based on trending detection
```

## Installation
We have updated the code to work with Python 3.8, Pytorch 1.7, and CUDA 10.2. If you use conda, you can set up the environment as follows:

```bash
conda create -n trending_NER python==3.8
conda activate trending_NER
conda install pytorch==1.7 cudatoolkit=10.2 -c pytorch
```

Also, install the dependencies specified in the requirements.txt:
```
pip install -r requirements.txt
```

## Data
Please download the data from: [Temporal Twitter Corpus](https://github.com/shrutirij/temporal-twitter-corpus).

Make sure you provide the correct paths to the data splits in the config file. 
For example, `exp_bert/configs/b2.0-bert-trend.json` contains this:

```json
    ...
    "partitions": {
        "train": "path/to/train.txt",
        "dev": "path/to/dev.txt",
        "test": [
            "path/to/test.txt"
        ]
    },
    ...
```

## Data Selection
For select the most informative data based on the trendig detection for retraining, please check `notebooks/1-exploring-trending-detection.ipynb` for detail.

## Running

This project contains two different systems:
1. Experiment with BERT and BERTweet
2. Experiment with CNN + LSTM +CRF

We use config files to specify the details for every experiment (e.g., hyper-parameters, datasets, etc.). You can modify config files in the `exp_bert/configs` directory or `exp_ssl/configs` directory and run experiments with following instructions.

### 1. Experiment with BERT and BERTweet

To run experiments with BERT, you can train the model from a config file like this:
```
CUDA_VISIBLE_DEVICES=1 python exp_bert/src/main.py --config exp_bert/configs/baseline/b2.0-bert-trend.json
```
To run experiments with BERTweet, you need to download pretrained weights with the following command (for more details, please check: [BERTweet](https://github.com/VinAIResearch/BERTweet)):
```
wget https://public.vinai.io/BERTweet_base_transformers.tar.gz
tar -xzvf BERTweet_base_transformers.tar.gz
```
Then you can modify the config files and train the model like this:

```
CUDA_VISIBLE_DEVICES=1 python exp_bert/src/main.py --config exp_bert/configs/baseline/b3.0-bertweet-trend.json
```

### 2. Experiment with CNN + LSTM +CRF
To run experiments with CNN + LSTM +CRF, you can train the model from a config file like this:
```
CUDA_VISIBLE_DEVICES=1 python exp_ssl/src/main.py --config exp_ssl/configs/baseline/b1.0-lstm-trend.json
```

## Citation
```text
@inproceedings{chen-etal-2021-mitigating,
    title = "Mitigating Temporal-Drift: A Simple Approach to Keep {NER} Models Crisp",
    author = "Chen, Shuguang and Neves, Leonardo and Solorio, Thamar",
    booktitle = "Proceedings of the Ninth International Workshop on Natural Language Processing for Social Media",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.socialnlp-1.14",
    doi = "10.18653/v1/2021.socialnlp-1.14",
    pages = "163--169"
}
```

## Contact
Feel free to get in touch via email to schen52@uh.edu.
