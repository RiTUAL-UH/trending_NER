import torch
import exp_bert.src.commons.utilities as utils

from typing import List
from emoji import demojize
from torch.utils.data import Dataset
from transformers import BertTokenizer

from fairseq.data import Dictionary
from fairseq.data.encoders.fastbpe import fastBPE


class NERDatasetBase(Dataset):
    def __init__(self,
                 dataset_file,
                 dataset_cols,
                 label_scheme: List[str],
                 tokenizer: BertTokenizer,
                 bpe: fastBPE,
                 vocab: Dictionary,
                 partition: str):

        self.tokenizer = tokenizer
        self.bpe = bpe
        self.vocab = vocab
        self.index_map = dict(enumerate(label_scheme))
        self.label_map = {l: i for i, l in self.index_map.items()}

        self.dataset_file = dataset_file
        self.dataset_cols = dataset_cols

        self.partition = partition

    def _init_data_fields(self, dataset=None):
        if dataset is None:
            dataset = utils.read_conll(self.dataset_file, columns=self.dataset_cols)
    
        self.tokens = dataset['tokens']
        self.labels = dataset['labels']

    def _prepare_encoding_fields_from_start(self, partition, use_tokenizer=True):
        """
        Only call this method if you want everything tokenized and encoded from the very beginning.
        This is not the case when there is dynamic masking.
        """
        datasets = utils.read_conll(self.dataset_file, columns=self.dataset_cols)

        self.tokenized = []
        self.input_ids = []
        self.label_ids = []
        self.label_msk = []

        for i in range(len(datasets['tokens'])):
            tokens = datasets['tokens'][i]
            labels = datasets['labels'][i]

            if use_tokenizer:
                tokenized, input_ids, label_ids, label_msk = process_sample_with_tokenizer(tokens, labels, self.label_map, self.tokenizer)
            else:
                tokenized, input_ids, label_ids, label_msk = process_sample_with_BPE(tokens, labels, self.label_map, self.bpe, self.vocab)
                
                # Remove long sentences. Maximum length is 130 for BERTweet.
                if len(input_ids) >= 130:
                    continue

            self.tokenized.append(tokenized)
            self.input_ids.append(input_ids)
            self.label_ids.append(label_ids)
            self.label_msk.append(label_msk)


    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()

    def collate(self, batch, pad_tok=0):
        raise NotImplementedError()


class NERDataset(NERDatasetBase):
    """
    This class encodes the data from the beginning.
    """
    def __init__(self,
                 dataset_file,
                 dataset_cols,
                 label_scheme: List[str],
                 tokenizer: BertTokenizer,
                 bpe: fastBPE,
                 vocab: Dictionary,
                 partition: str,
                 use_tokenizer=True):

        super().__init__(dataset_file, dataset_cols, label_scheme, tokenizer, bpe, vocab, partition)

        # Always encodes the data from the beginning, regardless the partition
        self._prepare_encoding_fields_from_start(partition=partition, use_tokenizer=use_tokenizer)

    def __len__(self):
        return len(self.tokenized)

    def __getitem__(self, index):
        input_ids = self.input_ids[index]
        label_ids = self.label_ids[index]
        label_msk = self.label_msk[index]

        return input_ids, label_ids, label_msk

    def collate(self, batch, pad_tok=0):
        # Unwrap the batch into every field
        input_ids, label_ids, label_msk = map(list, zip(*batch))

        # Padded variables
        p_input_ids, p_input_mask, p_token_type, p_label_ids, p_label_msk = [], [], [], [], []

        # How much padding do we need?
        max_seq_length = max(map(len, input_ids))

        for i in range(len(input_ids)):
            padding_length = max_seq_length - len(input_ids[i])

            p_input_ids.append(input_ids[i] + [pad_tok] * padding_length)
            p_input_mask.append([1] * len(input_ids[i]) + [pad_tok] * padding_length)
            p_token_type.append([0] * len(input_ids[i]) + [pad_tok] * padding_length)
            p_label_ids.append(label_ids[i] + [pad_tok] * padding_length)
            p_label_msk.append(label_msk[i] + [pad_tok] * padding_length)

        input_dict = {
            'input_ids': torch.tensor(p_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(p_input_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(p_token_type, dtype=torch.long),
            'label_mask': torch.tensor(p_label_msk, dtype=torch.long),
            'labels': torch.tensor(p_label_ids, dtype=torch.long)
        }

        return input_dict


def process_sample_with_tokenizer(tokens, labels, label_map, tokenizer):
    tokenized = []
    label_ids = []
    label_msk = []

    for i, (token, label) in enumerate(zip(tokens, labels)):
        word_tokens = tokenizer.tokenize(token)
        if len(word_tokens) == 0:
            word_tokens = [tokenizer.unk_token]
        num_subtoks = len(word_tokens) - 1

        tokenized.extend(word_tokens)
        label_ids.extend([label_map[label]] + [0] * num_subtoks)
        label_msk.extend([1] + [0] * num_subtoks)

    tokenized = [tokenizer.cls_token] + tokenized + [tokenizer.sep_token]

    input_ids = tokenizer.convert_tokens_to_ids(tokenized)
    label_ids = [0] + label_ids + [0]
    label_msk = [0] + label_msk + [0]

    return tokenized, input_ids, label_ids, label_msk


def process_sample_with_BPE(tokens, labels, label_map, bpe, vocab):
    tokenized = []
    label_ids = []
    label_msk = []

    for i, (token, label) in enumerate(zip(tokens, labels)):
        normalized = normalizeToken(token)
        word_tokens = bpe.encode(normalized).split()
        if len(word_tokens) == 0:
            word_tokens = ['mask']
        num_subtoks = len(word_tokens) - 1

        tokenized.extend(word_tokens)
        label_ids.extend([label_map[label]] + [0] * num_subtoks)
        label_msk.extend([1] + [0] * num_subtoks)

    subwords = '<s> ' + " ".join(tokenized) + ' </s>'
    input_ids = vocab.encode_line(subwords, append_eos=False, add_if_not_exist=False).long().tolist()
    label_ids = [0] + label_ids + [0]
    label_msk = [0] + label_msk + [0]

    return tokenized, input_ids, label_ids, label_msk


def normalizeToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return "@USER"
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "HTTPURL"
    elif len(token) == 1:
        return demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token

