import os
import re
import csv
import signal
import torch
import random
import numpy as np
from tabulate import tabulate
from itertools import groupby
from collections import defaultdict
from seqeval.metrics import f1_score, precision_score, recall_score
from seqeval.metrics import classification_report
from typing import List, Dict

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname((__file__)))))


def report2dict(cr):
    # Parse rows
    tmp = list()
    for row in cr.split("\n"):
        parsed_row = [x.strip() for x in row.split("  ") if len(x) > 0]
        if len(parsed_row) > 0:
            tmp.append(parsed_row)

    # Store in dictionary
    measures = tmp[0]

    D_class_data = defaultdict(dict)
    for row in tmp[1:]:
        class_label = row[0]
        for j, m in enumerate(measures):
            D_class_data[class_label][m.strip()] = float(row[j + 1].strip())
    return D_class_data


def printcr(report, classes=None, sort_by_support=False):
    headers = ['classes', 'precision', 'recall', 'f1-score', 'support']

    if classes is None:
        classes = [k for k in report.keys() if k not in {'macro avg', 'micro avg'}]

        if sort_by_support:
              classes = sorted(classes, key=lambda c: report[c]['support'], reverse=True)
        else: classes = sorted(classes)

    if 'macro avg' not in classes: classes.append('macro avg')
    if 'micro avg' not in classes: classes.append('micro avg')

    table = []
    for c in classes:
        if c == 'macro avg':
            table.append([])
        row = [c]
        for h in headers:
            if h not in report[c]:
                continue
            if h in {'precision', 'recall', 'f1-score'}:
                  row.append(report[c][h] * 100)
            else: row.append(report[c][h])
        table.append(row)
    print(tabulate(table, headers=headers, floatfmt=(".3f", ".3f", ".3f", ".3f")))
    print()


class EpochStats:
    def __init__(self):
        self.sizes = [] # number of elements per step
        self.losses = []

        self.probs = []
        self.preds = []
        self.golds = []

    def loss_step(self, loss: float, batch_size: int):
        self.losses.append(loss)
        self.sizes.append(batch_size)

    def step(self, scores, target, mask, loss):
        self.loss_step(loss, len(scores))

        probs, classes = scores.max(dim=2)

        for i in range(len(scores)):
            prob_i = probs[i][mask[i] == 1].cpu().tolist()
            pred_i = classes[i][mask[i] == 1].cpu().tolist()
            gold_i = target[i][mask[i] == 1].cpu().tolist()

            self.preds.append(pred_i) # self.preds.extend(pred_i)
            self.golds.append(gold_i) # self.golds.extend(gold_i)
            self.probs.append(prob_i) # self.probs.extend(prob_i)

    def loss(self, loss_type: str = ''):
        losses = self.losses
        return np.mean([l for l, s in zip(losses, self.sizes) for _ in range(s)]), np.min(losses), np.max(losses)

    def _map_to_labels(self, index2label):
        # Predictions should have been as nested list to separate predictions
        # Since we store the predictions across epochs during training, we need to wrap up this in a try except
        # so that it handles the flattened lists in case they are not nested. New runs will be nested
        try:
            golds = [[index2label[j] for j in i] for i in self.golds]
            preds = [[index2label[j] for j in i] for i in self.preds]
        except TypeError:
            golds = [index2label[i] for i in self.golds]
            preds = [index2label[i] for i in self.preds]
        return golds, preds

    def metrics(self, index2label: [List[str], Dict[int, str]]):
        golds, preds =self._map_to_labels(index2label)

        f1 = f1_score(golds, preds)
        p = precision_score(golds, preds)
        r = recall_score(golds, preds)

        return f1, p, r

    def get_classification_report(self, index2label: [List[str], Dict[int, str]]):
        golds, preds = self._map_to_labels(index2label)

        cr = classification_report(golds, preds, digits=5)
        return report2dict(cr)

    def print_classification_report(self, index2label: [List[str], Dict[int, str]] = None, report = None):
        assert index2label is not None or report is not None

        if report is None:
            report = self.get_classification_report(index2label)

        printcr(report)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_conll(filename, columns, delimiter='\t'):
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
                row = [data[col][sample_i][token_i] for col in colnames]
                fp.write(delimiter.join(row) + '\n')
            fp.write('\n')


def read_conll_corpus(corpus_dir, filenames, columns, delimiter='\t'):
    corpus = {}
    for datafile in filenames:
        dataset = os.path.splitext(datafile)[0]
        datafile = os.path.join(corpus_dir, datafile)
        corpus[dataset] = read_conll(datafile, columns, delimiter=delimiter)
    return corpus


def flatten(nested_elems):
    return [elem for elems in nested_elems for elem in elems]


def replace_numbers(token):
    if not re.search('[a-zA-Z]', token):
        return re.sub('\d', '#', token)
    return token


def print_popular_lm_parameters():
    from allennlp.modules.elmo import Elmo
    from flair.embeddings import FlairEmbeddings
    from transformers import BertModel

    elmo_weights = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"
    elmo_options = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
    elmo = Elmo(elmo_options, elmo_weights, 2)

    flair_lm = FlairEmbeddings('news-forward').lm

    bert_base_cased = BertModel.from_pretrained("bert-base-cased")
    bert_large_cased = BertModel.from_pretrained("bert-large-cased")

    def count_parameters(model, give_me_all):
        return sum(p.numel() for p in model.parameters() if p.requires_grad or give_me_all)

    print("elmo:", count_parameters(elmo, True))  # 93,600,872
    print("flair_lm:", count_parameters(flair_lm, True))  # 18,257,500
    print("bert_base_cased:", count_parameters(bert_base_cased, True))  # 108,310,272
    print("bert_large_cased:", count_parameters(bert_large_cased, True))  # 333,579,264


def input_with_timeout(prompt, timeout, default=''):
    def alarm_handler(signum, frame):
        raise Exception("Time is up!")
    try:
        # set signal handler
        signal.signal(signal.SIGALRM, alarm_handler)
        signal.alarm(timeout)  # produce SIGALRM in `timeout` seconds

        return input(prompt)
    except Exception as ex:
        return default
    finally:
        signal.alarm(0)  # cancel alarm

