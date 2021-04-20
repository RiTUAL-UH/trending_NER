import torch
import torch.nn as nn

from allennlp.modules import ConditionalRandomField


class InferenceLayer(nn.Module):
    def __init__(self, input_dim, n_classes, use_crf):
        super(InferenceLayer, self).__init__()

        self.use_crf = use_crf
        self.input_dim = input_dim
        self.output_dim = n_classes

        self.proj = nn.Linear(input_dim, n_classes)

        if self.use_crf:
            self.crf = ConditionalRandomField(n_classes, constraints=None, include_start_end_transitions=True)
        else:
            self.xent = nn.CrossEntropyLoss(reduction='mean')

    def crf_forward(self, logits, mask, target):
        mask = mask.long()
        best_paths = self.crf.viterbi_tags(logits, mask)
        tags, viterbi_scores = zip(*best_paths)
        loss = -self.crf.forward(logits, target, mask)  # neg log-likelihood loss
        loss = loss / torch.sum(mask)

        return {'loss': loss, 'logits': logits, 'tags': tags, 'path_scores': viterbi_scores}

    def fc_forward(self, logits, mask, target):
        assert len(logits.size()) == 3

        if mask is not None:
            mask = mask.long()
            tags = torch.softmax(logits, dim=2).max(-1)
            tags = tags[1].cpu().tolist()

            for i in range(len(tags)):
                tags[i] = tags[i][:mask[i].sum().item()]

            mask = mask.view(-1) == 1

            logits_ = logits.view(-1, logits.size(-1))
            target_ = target.view(-1)

            loss = self.xent(logits_[mask], target_[mask])
        else:
            tags = torch.softmax(logits, dim=2).max(-1)
            tags = tags[1].cpu().tolist()

            for i in range(len(tags)):
                tags[i] = tags[i][:]

            logits_ = logits.view(-1, logits.size(-1))
            target_ = target.view(-1)

            loss = self.xent(logits_, target_)

        return {'loss': loss, 'logits': logits, 'tags': tags}

    def forward(self, vectors, mask, targets):
        logits = self.proj(vectors)

        if self.use_crf:
            results = self.crf_forward(logits, mask, targets)
        else:
            results = self.fc_forward(logits, mask, targets)

        results['mask'] = mask.data if mask is not None else None

        return results


class LSTMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, bidirectional, num_layers, drop_prob=0.3):
        super(LSTMLayer, self).__init__()

        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim // 2 if bidirectional else hidden_dim,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            dropout=drop_prob if num_layers > 1 else 0,
                            batch_first=True)

    def forward(self, vectors, mask, hidden=None):
        batch_size = vectors.size(0)
        max_length = vectors.size(1)
        lengths = mask.view(batch_size, max_length).long().sum(-1)

        if hidden is not None:
            directions = 2 if self.lstm.bidirectional else 1
            (h0, c0) = hidden  # (num_layers * num_directions, batch, hidden_size)
            assert h0.size(0) == c0.size(0) == self.num_layers * directions
            assert h0.size(1) == c0.size(1) == batch_size
            assert h0.size(2) == c0.size(2) == self.hidden_dim // directions

        lstm_outs, _ = self.lstm(vectors, hidden)  # (batch, seq_len, num_directions * hidden_size)

        assert lstm_outs.size(0) == batch_size
        assert lstm_outs.size(1) == max_length
        assert lstm_outs.size(2) == self.hidden_dim

        if self.bidirectional:
            # Separate the directions of the LSTM
            lstm_outs = lstm_outs.view(batch_size, max_length, 2, self.hidden_dim // 2)

            # Pick up the last hidden state per direction
            fw_last_hn = lstm_outs[range(batch_size), lengths - 1, 0]   # (batch, hidden // 2)
            bw_last_hn = lstm_outs[range(batch_size), 0, 1]             # (batch, hidden // 2)

            lstm_outs = lstm_outs.view(batch_size, max_length, self.hidden_dim)

            last_hn = torch.cat([fw_last_hn, bw_last_hn], dim=1)        # (batch, hidden // 2) -> (batch, hidden)
        else:
            last_hn = lstm_outs[range(batch_size), lengths - 1]         # (batch, hidden)

        return {'last': last_hn, 'outputs': lstm_outs}


class CNNLayer(nn.Module):
    def __init__(self, input_dim, channels, kernels, maxlen):
        super(CNNLayer, self).__init__()

        assert len(kernels) == len(channels)

        self.input_dim = input_dim
        self.maxlen = maxlen  # maximum sequence length
        self.kernels = kernels  # playing the role of n-gram of different orders
        self.channels = channels  # the number of output channels per convolution layer
        self.output_dim = sum(self.channels)

        self.cnn = {}
        self.bn = {}

        for kernel, out_channels in zip(kernels, channels):
            self.cnn[f'{kernel}_gram'] = nn.Conv1d(self.input_dim, out_channels, kernel)
            self.bn[f'{kernel}_gram'] = nn.BatchNorm1d(out_channels)

        self.cnn = nn.ModuleDict(self.cnn)
        self.bn = nn.ModuleDict(self.bn)

    def forward(self, embeddings):
        batch_size = embeddings.size(0)
        seq_length = embeddings.size(1)
        seq_maxlen = min(seq_length, self.maxlen)

        # Prepare for sliding the Conv1d across time
        embeddings = embeddings.transpose(1, 2)  # -> (batch, embedding, seq_length)

        convs = []
        for kernel, channels in zip(self.kernels, self.channels):
            cnn_key = f'{kernel}_gram'

            convolved = self.cnn[cnn_key](embeddings)  # -> (batch, n_filters, channels)

            curr_shape = convolved.size()
            expt_shape = (batch_size, channels, seq_maxlen - kernel + 1)

            assert curr_shape == expt_shape, "Wrong size: {}. Expected {}".format(curr_shape, expt_shape)

            convolved = self.bn[cnn_key](convolved)  # -> (batch, n_filters, channels)
            convolved, _ = torch.max(convolved, dim=2)  # -> (batch, n_filters)
            convolved = torch.nn.functional.relu(convolved)
            convs.append(convolved)

        convs = torch.cat(convs, dim=1)  # -> (batch, sum(n_filters))  dim 1 is the sum of n_filters from all cnn layers

        return {'outputs': convs}

