import os
import csv
import torch
from itertools import groupby
from typing import List, Dict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from collections import Counter
from sklearn.utils import shuffle


def flatten(l):
    return [i for sublist in l for i in sublist]


def save_predictions(filepath, samples, truth, preds, scores):
    assert len(samples) == len(truth) == len(preds) == len(scores)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w') as fp:
        for i in range(len(samples)):
            assert len(samples[i]) == len(truth[i]) == len(preds[i]) == len(scores[i])

            for j in range(len(samples[i])):
                fp.write("{}\t{}\t{}\t{:.5f}\n".format(
                    samples[i][j], truth[i][j], preds[i][j], scores[i][j]))
            fp.write('\n')


def save_path_scores(filepath, scores):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as fp:
        for i in range(len(scores)):
            fp.write('{:.5f}\n'.format(scores[i]))


def process_logits(logits, tags):
    probs = []
    logits = torch.softmax(logits.data, dim=-1)

    for i, instance_tags in enumerate(tags):
        instance_probs = []
        for j, tag_id in enumerate(instance_tags):
            instance_probs.append(logits[i, j, tag_id].item())
        probs.append(instance_probs)

    return probs


def count_params(model):
    return sum([p.nelement() for p in model.parameters() if p.requires_grad])


def get_dataloader(dataset, batch_size, shuffle=False):
    sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
    dloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=dataset.collate_fn)
    return dloader


def create_dataloaders(datasets, args):
    oarg = args.training.optim
    train_batch_size = args.training.batch_size * max(1, oarg.n_gpu)
    eval_batch_size = args.evaluation.batch_size * (max(1, oarg.n_gpu))
    dataloaders = {
        'train': get_dataloader(datasets['train'], batch_size=train_batch_size, shuffle=True),
        'dev':   get_dataloader(datasets['dev'],   batch_size=eval_batch_size, shuffle=False),
        'test':  get_dataloader(datasets['test'],  batch_size=eval_batch_size, shuffle=False)
    }
    return dataloaders


def get_label_to_index(datasets):
    all_labels = flatten([flatten(datasets[dataset].labels) for dataset in datasets])
    all_labels = {l: i for i, l in enumerate(sorted(set(all_labels)))}
    return all_labels


def read_conll(filename, columns: List[str], delimiter='\t'):
    def is_empty_line(line_pack):
        return all(field.strip() == '' for field in line_pack)

    data = []
    with open(filename) as fp:
        reader = csv.reader(fp, delimiter=delimiter, quoting=csv.QUOTE_NONE)
        groups = groupby(reader, is_empty_line)

        for is_empty, pack in groups:
            if is_empty is False:
                data.append([list(field) for field in zip(*pack)])

    data = list(zip(*data))
    dataset = {colname: list(data[columns[colname]]) for colname in columns}

    return dataset


def write_conll(filename, data, colnames: List[str] = None, delimiter='\t'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if colnames is None:
        colnames = list(data.keys())

    any_key = colnames[0]

    with open(filename, 'w') as fp:
        for sample_i in range(len(data[any_key])):
            for token_i in range(len(data[any_key][sample_i])):
                row = [str(data[col][sample_i][token_i]) for col in colnames]
                fp.write(delimiter.join(row) + '\n')
            fp.write('\n')


def print_frequent_hashtags(datasets):
    hashtags = []
    for split in datasets:
        dtokens = datasets[split].tokens
        for tokens in dtokens:
            for i, token in enumerate(tokens):
                if token == '#' and i + 1 < len(tokens):
                    hashtags.append('#' + tokens[i+1])
                if token.startswith('#') and token != '#':
                    hashtags.append(token)
    
    hashtags = Counter(hashtags).most_common(10)
    # hashtags = [item for item, num in hashtags]
    print(hashtags)


def shuffle_datasets(datasets, num_samples=2200, ratio=[0.7, 0.15, 0.15]):
    tokens, labels = [], []
    for split in datasets:
        if split == 'test':
            continue
        tokens.extend(datasets[split].tokens)
        labels.extend(datasets[split].labels)
    
    tokens, labels = shuffle(tokens, labels)

    datasets['train'].tokens, datasets['train'].labels = tokens[0: 1540], labels[0: 1540]
    datasets['dev'].tokens, datasets['dev'].labels = tokens[1540: 1870], labels[1540: 1870]
    datasets['test'].tokens, datasets['test'].labels = tokens[1870: 2200], labels[1870: 2200]

    return datasets


def reform_test_by_hashtags(dataset, hashtags):
    new_tokens, new_labels = [], []
    for i in range(len(dataset.tokens)):
        tokens = dataset.tokens[i]
        labels = dataset.labels[i]
        intersections = list(set(hashtags) & set(tokens))
        if len(intersections) > 0:
            new_tokens.append(tokens)
            new_labels.append(labels)
    
    dataset.tokens, dataset.labels = new_tokens, new_labels

    return dataset

