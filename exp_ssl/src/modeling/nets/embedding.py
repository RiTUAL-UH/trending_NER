import re
import time
import emoji
import string
import torch
import validators
import numpy as np
import torch.nn as nn
import exp_ssl.src.commons.globals as glb

from typing import List, Tuple
from flair.embeddings import WordEmbeddings


def get_word_vectors(embedding, scale='none'):
    assert embedding in {'twitter', 'glove', 'crawl'}
    model = WordEmbeddings(embedding).precomputed_word_embeddings

    vectors = model.vectors

    if scale == 'z-standardization':
        print("[LOG] Scaling embeddings using {}".format(scale))
        mu = vectors.mean(axis=0)
        sigma = vectors.std(axis=0)
        vectors = (vectors - mu) / sigma

    elif scale == 'normalization':
        print("[LOG] Scaling embeddings using {}".format(scale))
        vectors = vectors / np.linalg.norm(vectors, ord=2, axis=1, keepdims=True)

    elif scale == 'scale(-1,1)':
        print("[LOG] Scaling embeddings using {}".format(scale))
        min_vector = vectors.min(axis=0)
        max_vector = vectors.max(axis=0)

        min_target = -1
        max_target = 1

        vectors = ((vectors - min_vector) / (max_vector - min_vector)) * (max_target - min_target) + min_target
    else:
        print("[LOG] Embeddings are not scaled and will be loaded as-is")


    return model.index2word, vectors


def get_gaussian_sampler(v):
    mean = torch.from_numpy(v.mean(axis=0))
    std = torch.from_numpy(v.std(axis=0))
    normal = torch.distributions.normal.Normal(mean, std)
    return normal


def convert_word(word, word2index):
    if word.startswith('@') and 2 < len(word) <= 16 and all(str.isalpha(char) or str.isnumeric(char) or char == '_' for char in word[1:]):
        return '<user>'

    elif str.isnumeric(word) or str.isdigit(word) or str.isdecimal(word) or re.match("^\d+?\.\d+?$", word) is not None:
        return '<number>'

    elif validators.url(word) is True or validators.domain(word) is True or validators.email(word) is True:
        return '<url>'

    elif word in emoji.UNICODE_EMOJI or all(char in emoji.UNICODE_EMOJI for char in word):
        return '<emoji>'

    elif all(char in string.punctuation for char in word):
        return '<punctuation>'

    elif word.lower() in word2index:
        return word.lower()

    elif word.title() in word2index:
        return word.title()

    elif word.upper() in word2index:
        return word.upper()

    elif re.sub(r'\d', '#', word.lower()) in word2index:
        return re.sub(r'\d', '#', word.lower())

    elif re.sub(r'\d', '0', word.lower()) in word2index:
        return re.sub(r'\d', '0', word.lower())

    else:
        return None


class TokenEmbedder(nn.Module):
    def __init__(self, vocab: List[Tuple[str, int]], embeddings: List[str]):
        super(TokenEmbedder, self).__init__()

        print("[LOG] Creating TokenEmbedder layer")
        start_time = time.time()

        self.UNK = '<unk>'
        self.PAD = '<pad>'

        precomp_index2word, precomp_vectors = get_word_vectors(embeddings.pop(), scale="none") # "scale='z-standardization')
        precomp_index2word = dict(enumerate(precomp_index2word))
        precomp_word2index = {word: i for i, word in precomp_index2word.items()}

        gaussian = get_gaussian_sampler(precomp_vectors)

        self.embedding_length = precomp_vectors.shape[1]

        self.index_to_word = [self.PAD, self.UNK]
        self.vectors = [torch.zeros(1, self.embedding_length), gaussian.sample().view(1, self.embedding_length)]

        token_set = set(self.index_to_word) # it's faster to check on a set than the 'index_to_word' list

        count_oov = 0  # out-of-vocabulary count
        count_inv = 0  # in-vocabulary count
        count_ign = 0  # ignored count
        count_rnd = 0  # random count
        count_cvt = 0  # converted count

        for word, freq in vocab:
            if word not in precomp_word2index:
                count_oov += 1

                word = convert_word(word, precomp_word2index)

                if word is None:
                    count_ign += 1
                    continue

                if word not in precomp_word2index:
                    count_rnd += 1
                    vector = gaussian.sample()
                else:
                    count_cvt += 1
                    vector = torch.from_numpy(precomp_vectors[precomp_word2index[word]])
            else:
                count_inv += 1
                vector = torch.from_numpy(precomp_vectors[precomp_word2index[word]])

            if word not in token_set:
                self.vectors.append(vector.view(1, self.embedding_length))
                self.index_to_word.append(word)
                token_set.add(word)


        print("[LOG] It took {:.1f}s to filter the vocabulary...".format(time.time() - start_time))
        print("[LOG]          Vocabulary size: {:,}".format(len(vocab)))
        print("[LOG]  Out-of-vocabulary count: {:,}".format(count_oov))
        print("[LOG]      In-vocabulary count: {:,}".format(count_inv))
        print("[LOG]          Converted count: {:,}".format(count_cvt))
        print("[LOG]      Ignored words count: {:,}".format(count_ign))
        print("[LOG] Random initialized count: {:,}".format(count_rnd))

        self.vectors = nn.Embedding.from_pretrained(torch.cat(self.vectors, dim=0), freeze=False)
        self.index_to_word = dict(enumerate(self.index_to_word))
        self.word_to_index = {w: i for i, w in self.index_to_word.items()}


    def forward(self, sentences):
        batch_size = len(sentences)
        seq_length = max([len(s) for s in sentences])

        mask = torch.zeros(batch_size, seq_length).long()
        encs = torch.zeros(batch_size, seq_length).long()

        UNK_IX = self.word_to_index[self.UNK]

        for i, s in enumerate(sentences):
            encs[i, :len(s)] = torch.tensor([self.word_to_index.get(convert_word(word, self.word_to_index), UNK_IX) for word in s], dtype=torch.long)
            mask[i, :len(s)] = 1

        result = {
            'outputs': self.vectors(encs.to(glb.DEVICE)),
            'mask': mask.to(glb.DEVICE)
        }

        return result


class CharEmbedder(nn.Module):
    def __init__(self, alphabet: List[str], embedding_length):
        super(CharEmbedder, self).__init__()

        self.UNK = '<unk>'
        self.PAD = '<pad>'
        self.MAX = 16

        self.embedding_length = embedding_length
        self.index_to_char = dict(enumerate([self.PAD, self.UNK] + sorted(set(alphabet))))
        self.char_to_index = {char: index for index, char in self.index_to_char.items()}

        self.embedding = nn.Embedding(len(self.index_to_char), embedding_length)

    def forward(self, sentences):
        seq_length = max([len(s) for s in sentences])
        chr_length = min(max([len(t) for s in sentences for t in s]), self.MAX)
        
        mask = []
        inputs = []
        for i, s in enumerate(sentences):
            for j, t in enumerate(s + [self.PAD] * (seq_length - len(s))):
                if t == self.PAD:
                    inputs.append([self.PAD] * chr_length)
                    mask.append([0] * chr_length)
                else:
                    allowed = t[:chr_length]
                    padding = [self.PAD] * (chr_length - len(t))
                    inputs.append(list(allowed) + padding)
                    mask.append([1] * len(allowed) + [0] * len(padding))

        encodings = torch.tensor([[self.char_to_index.get(char, self.char_to_index[self.UNK]) for char in chars] for chars in inputs]).long()

        result = {
            'outputs': self.embedding(encodings.to(glb.DEVICE)),
            'mask': torch.tensor(mask, dtype=torch.long).to(glb.DEVICE)
        }

        return result

