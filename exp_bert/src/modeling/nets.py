import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, AutoModel

from exp_bert.src.modeling.layers import InferenceLayer


class NERModelBase(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

        # only load configuration without weights
        self.bert = AutoModel.from_config(config)

        self.classifier = InferenceLayer(config.hidden_size, config.num_labels, use_crf=True)
        
        if config.pretrained_frozen:
            print("Freezing BERT parameters")
            for param in self.bert.parameters():
                param.requires_grad = False

        if config.vocab_size != self.bert.config.vocab_size:
            self.bert.resize_token_embeddings(config.vocab_size)

            if config.pretrained_frozen:
                print("[WARNING] New tokens have been added, but BERT won't be trainable")

    def forward_bert(self, input_ids, attention_mask, token_type_ids, position_ids=None, head_mask=None, inputs_embeds=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        if self.config.output_attentions:
            attentions = outputs[-1]
            return sequence_output, pooled_output, attentions
        else:
            return sequence_output, pooled_output

    def ner_loss(self, logits, labels, label_mask=None):
        if labels is None:
            loss = torch.tensor(0, dtype=torch.float, device=logits.device)
        else:
            # Only keep active parts of the loss
            if label_mask is not None:
                    loss_fct = nn.CrossEntropyLoss()
                    active_loss = label_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)[active_loss]
                    active_labels = labels.view(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss

    def forward(self, input_ids, attention_mask, token_type_ids,
                position_ids=None, labels=None, label_mask=None, head_mask=None, inputs_embeds=None,
                wrap_scalars=False):
        raise NotImplementedError('The NERModelBase class should never execute forward')


class NERModel(NERModelBase):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_ids, attention_mask, token_type_ids,
                position_ids=None, labels=None, label_mask=None, head_mask=None, inputs_embeds=None,
                wrap_scalars=False):

        sequence_output, _ = self.forward_bert(input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds)

        loss, logits = self.classifier(sequence_output, attention_mask, labels)

        if wrap_scalars: # for parallel models
            loss = loss.unsqueeze(0)

        return loss, logits

