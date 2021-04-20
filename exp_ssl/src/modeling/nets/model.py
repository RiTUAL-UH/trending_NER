import torch
import torch.nn as nn
import h5py

from exp_ssl.src.modeling.nets.layers import LSTMLayer, CNNLayer, InferenceLayer
from exp_ssl.src.modeling.nets.embedding import TokenEmbedder, CharEmbedder
from exp_ssl.src.commons import globals as glb
from typing import List, Tuple


class NERModelBase(nn.Module):
    def __init__(self,
                 alphabet: List[str],
                 vocab: List[Tuple[str, int]],
                 char_embedding_dim: int,
                 char_kerners: List[int],
                 char_channels: List[int],
                 embeddings: List[str],
                 lstm_size: int,
                 lstm_layers: int,
                 bidirectional: bool,
                 use_crf: bool,
                 dropout: float,
                 n_classes: InterruptedError):

        super().__init__()

        self.dropout = nn.Dropout(dropout)

         # Word Feats
        self.char_emb = CharEmbedder(alphabet, char_embedding_dim)
        self.char_cnn = CNNLayer(char_embedding_dim, channels=char_channels, kernels=char_kerners, maxlen=self.char_emb.MAX)
        self.word_emb = TokenEmbedder(vocab, embeddings)

        # BiLSTM
        self.lstm = LSTMLayer(input_dim=self.word_emb.embedding_length + self.char_cnn.output_dim,
                            hidden_dim=lstm_size,
                            bidirectional=bidirectional,
                            num_layers=lstm_layers,
                            drop_prob=dropout)


    def getWordFeats(self, inputs):
        # word embedding
        word_emb_res = self.word_emb(inputs)

        batch_size, seq_length, _ = word_emb_res['outputs'].size()

        # char embedding
        char_emb_res = self.char_emb(inputs)

        # char CNN
        char_cnn_res = self.char_cnn(char_emb_res['outputs'])
        char_cnn_res['outputs'] = char_cnn_res['outputs'].view(batch_size, seq_length, -1)
        char_cnn_res['outputs'] = self.dropout(char_cnn_res['outputs'])

        # [word representation, character representation]
        word_feats = torch.cat((word_emb_res['outputs'], char_cnn_res['outputs']), dim=-1)

        return word_feats, word_emb_res['mask']


    def forward_lstm(self, inputs, targets):
        # Word Feats
        word_feats, mask = self.getWordFeats(inputs)
        word_feats = self.dropout(word_feats)

        # BiLSTM
        lstm_results = self.lstm(word_feats, mask)

        return lstm_results, mask


    def forward(self, inputs, targets, trends=None, years=None):
        raise NotImplementedError('The NERModelBase class should never execute forward')


class NERModel(NERModelBase):
    def __init__(self,
                 alphabet: List[str],
                 vocab: List[Tuple[str, int]],
                 char_embedding_dim: int,
                 char_kerners: List[int],
                 char_channels: List[int],
                 embeddings: List[str],
                 lstm_size: int,
                 lstm_layers: int,
                 bidirectional: bool,
                 use_crf: bool,
                 dropout: float,
                 n_classes: InterruptedError):

        super().__init__(alphabet, vocab, char_embedding_dim, char_kerners, char_channels, embeddings, lstm_size, lstm_layers, bidirectional, use_crf, dropout, n_classes)
        
        # CRF
        self.classifier = InferenceLayer(lstm_size, n_classes, use_crf=use_crf)

    def forward(self, inputs, targets, trends=None, years=None):
        # LSTM
        lstm_results, mask = self.forward_lstm(inputs, targets)

        # CRF
        clf_results  = self.classifier(lstm_results['outputs'], mask, targets)

        return clf_results

