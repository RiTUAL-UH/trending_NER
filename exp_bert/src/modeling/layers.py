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
        loss = -self.crf.forward(logits, target, mask)  # neg log-likelihood loss
        loss = loss / torch.sum(mask)

        return loss, logits

    def fc_forward(self, logits, mask, target):
        if mask is not None:
            mask = mask.long()
            mask = mask.view(-1) == 1

            logits_ = logits.view(-1, logits.size(-1))[mask]
            target_ = target.view(-1)[mask]
            loss = self.xent(logits_, target_)
        else:
            loss = self.xent(logits.view(-1, logits.size(-1)), target.view(-1))

        return loss, logits

    def forward(self, vectors, mask, targets):
        logits = self.proj(vectors)

        if self.use_crf:
            loss, logits = self.crf_forward(logits, mask, targets)
        else:
            loss, logits = self.fc_forward(logits, mask, targets)

        return loss, logits

