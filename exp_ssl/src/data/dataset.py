import os
import torch
from torch.utils.data import Dataset

import exp_ssl.src.commons.globals as glb
import exp_ssl.src.commons.utilities as utils


class SMDataset(Dataset):
    def __init__(self, datadir, colnames):
        assert os.path.exists(datadir), datadir

        dataset = utils.read_conll(datadir, colnames)

        self.tokens = dataset['tokens']
        self.labels = dataset['labels']

        if 'trends' in dataset:
            self.trends = dataset['trends']
        else:
            self.trends = [[0 for token in tokens] for tokens in self.tokens]
        
        if 'years' in dataset:
            self.years = dataset['trends']
        else:
            self.years = [[2014 for token in tokens] for tokens in self.tokens]

        assert len(self.tokens) == len(self.labels) == len(self.trends) == len(self.years)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, item):
        return self.tokens[item], self.labels[item], self.trends[item], self.years[item]

    def collate_fn(self, batch):
        tokens, labels, trends, years = zip(*batch)

        batch_size = len(tokens)
        seq_maxlen = max(list(map(len, tokens)))

        ner_targets = torch.zeros(batch_size, seq_maxlen).long().to(glb.DEVICE)
        trend_target = torch.zeros(batch_size, seq_maxlen).long().to(glb.DEVICE)
        year_target = torch.zeros(batch_size).long().to(glb.DEVICE)

        for i in range(batch_size):
            for j in range(len(tokens[i])):
                ner_targets[i, j] = labels[i][j]
                trend_target[i, j] = int(trends[i][j])
            year_target[i] = int(years[i][0]) - 2014

        batch_dict = {
            'tokens': tokens,
            'targets': ner_targets,
            'trends': trend_target,
            'years': year_target
        }

        return batch_dict

    def encode_labels(self, label_to_index):
        self.labels = map_terms(self.labels, label_to_index)

    def decode_labels(self, index_to_labels):
        self.labels = map_terms(self.labels, index_to_labels)

    def save_conll(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as fp:
            for i in range(len(self.tokens)):
                for j in range(len(self.tokens[i])):
                    fp.write("{}\t{}\n".format(self.tokens[i][j], self.labels[i][j]))
                fp.write('\n')


def create_datasets(train_path, dev_path, test_path):
    datasets = {
        'train': SMDataset(train_path, colnames={'tokens': 0, 'labels': 1, 'trends': 2, 'years': 3}),
        'dev':   SMDataset(dev_path, colnames={'tokens': 0, 'labels': 1}),
        'test':  SMDataset(test_path, colnames={'tokens': 0, 'labels': 1})
    }
    return datasets


def map_terms(terms, mapper):

    for i in range(len(terms)):
        for j in range(len(terms[i])):
            terms[i][j] = mapper[terms[i][j]]
    return terms
