
import os, sys
import numpy as np
import torch
import torch.optim as optim
from torch import autograd
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from collections import Counter
from copy import deepcopy

import exp_ssl.src.commons.utilities as utils
import exp_ssl.src.commons.globals as glb
import exp_ssl.src.modeling.nets.model as nets
from exp_ssl.src.data.dataset import create_datasets
from exp_ssl.src.modeling.train import train, evaluate


def choose_model(args, alphabet, vocab, n_classes):
    if args.name == 'NERModel':
        model = nets.NERModel(alphabet,
                                vocab,
                                args.char_model.embedding_dim,
                                args.char_model.kernels,
                                args.char_model.channels,
                                args.embeddings,
                                args.lstm_hidden,
                                args.lstm_layers,
                                args.lstm_bidirectional,
                                args.use_crf,
                                args.dropout,
                                n_classes)

    else:
        raise NotImplemented
    
    return model


def choose_optimizer(model, args):
    params = filter(lambda p: p.requires_grad, model.parameters())

    if args.name == 'sgd':
        optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum)
    elif args.name == 'adam':
        optimizer = optim.Adam(params, lr=args.lr)
    else:
        raise NotImplementedError('Optimizer not implemented: {}'.format(args.name))

    return optimizer


def choose_lr_scheduler(optimizer, args):
    if args.lr_scheduler.name == 'cos':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.lr_scheduler.T_0, T_mult=args.lr_scheduler.T_mult)
    else:
        scheduler = None
        
    return scheduler


def load_model(model, checkpoint_path):
    state = torch.load(checkpoint_path, map_location=glb.DEVICE)
    model.load_state_dict(state['model'])

    if 'f1' in state:
          f1 = state['f1']
    else: f1 = state['acc']  # the F1 got saved as acc on some models TODO: remove this if

    print("[LOG] Loaded checkpoint from global_step {} with loss {:.4f} and F1 {:.4f}".format(state['global_step'], state['loss'], f1))
    return model


def main(args):
    datasets = create_datasets(args.dataset.train, args.dataset.dev, args.dataset.test)

    # vocab = Counter(utils.flatten(datasets['train'].tokens + datasets['dev'].tokens)).most_common()
    vocab = torch.load('data/vocab.th')
    alphabet = sorted(set(utils.flatten([word for word, _ in vocab])))

    label_to_index = utils.get_label_to_index(datasets)
    
    print("[LOG] Train data size: {:,}".format(len(datasets['train'])))
    print("[LOG]   Dev data size: {:,}".format(len(datasets['dev'])))
    print("[LOG]  Test data size: {:,}".format(len(datasets['test'])))

    datasets['train'].encode_labels(label_to_index)
    datasets['dev'].encode_labels(label_to_index)
    datasets['test'].encode_labels(label_to_index)

    dataloader = utils.create_dataloaders(datasets, args)

    print("[LOG] " + "=" * 40)
    print("[LOG]", label_to_index)
    print("[LOG] " + "=" * 40)

    model = choose_model(args.model, alphabet, vocab, len(label_to_index))
    model.to(glb.DEVICE)

    print("[LOG] " + "=" * 40)
    print("[LOG] Parameter count: {}".format(utils.count_params(model)))
    print("[LOG] " + "=" * 40)

    checkpoint_path = os.path.join(args.checkpoints, 'model.pt')

    if args.mode == 'train':
        if os.path.exists(checkpoint_path):
            option = input("[LOG] Found a checkpoint! Choose an option:\n"
                           "\t0) Train from scratch and override the previous checkpoint\n"
                           "\t1) Load the checkpoint and train from there\nYour choice: ")
            assert option in {"0", "1"}, "Unexpected choice"

            if option == "1":
                model = load_model(model, checkpoint_path)

        optimizer = choose_optimizer(model, args.training.optim)
        scheduler = choose_lr_scheduler(optimizer, args.training)
        model = train(model, dataloader, optimizer, scheduler, label_to_index, args)

    elif args.mode == 'eval':
        model = load_model(model, checkpoint_path)

    evaluate(model, dataloader, label_to_index, args)

