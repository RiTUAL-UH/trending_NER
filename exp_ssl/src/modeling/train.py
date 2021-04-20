
import os
import torch
import torch.nn as nn

from tqdm import tqdm, trange
from seqeval.metrics import f1_score, classification_report
from exp_ssl.src.commons.utilities import process_logits, save_predictions, save_path_scores


def decode_labels(label_to_index, encoded_labels):
    index_to_label = {index: label for label, index in label_to_index.items()}

    for i in range(len(encoded_labels)):
        for j in range(len(encoded_labels[i])):
            encoded_labels[i][j] = index_to_label[encoded_labels[i][j]]

    return encoded_labels


def track_best_model(model_path, model, best_f1, best_step, dev_f1, dev_loss, global_step):
    if best_f1 > dev_f1:
        return best_f1, best_step
    state = {
        'f1': dev_f1,
        'loss': dev_loss,
        'model': model.state_dict(),
        'global_step': global_step
    }
    os.makedirs(model_path, exist_ok=True)
    torch.save(state, os.path.join(model_path, 'model.pt'))
    return dev_f1, global_step


def print_stats(stats):
    print()
    for i in range(len(stats['train']['loss'])):
        epoch_msg = f'Epoch {i+1:03d}'
        for split in stats:
            epoch_msg += ' [{}] Loss: {:.4f}, F1: {:.4f}'.format(split.upper(), stats[split]['loss'][i], stats[split]['f1'][i])
        print(epoch_msg)
    print()


def evaluate(model, dataloaders, label_to_index, args):

    for dataset in dataloaders:

        if dataset == 'train':
            continue
        
        model.eval()

        dataset_tokens = 0
        dataset_loss = 0
        samples, preds, truth, scores, path_scores = [], [], [], [], []

        print("[LOG] " + "=" * 40)
        print("[LOG] {} Dataset".format(dataset.title()))
        print("[LOG] " + "=" * 40)

        # ========================================================================
        for batch_dict in tqdm(dataloaders[dataset]):

            tokens, targets = batch_dict['tokens'], batch_dict['targets']

            result = model(tokens, targets)

            dataset_tokens += torch.sum(result['mask']).item()
            dataset_loss += result['loss'].item() * torch.sum(result['mask']).item()
            batch_preds = result['tags']

            samples += tokens
            preds += batch_preds
            truth += [t[:len(tokens[i])] for i, t in enumerate(targets.data.cpu().tolist())]
            scores += process_logits(result['logits'], result['tags'])
            if 'path_scores' in result:
                path_scores += result['path_scores']
        # ========================================================================

        decoded_truth = decode_labels(label_to_index, truth)
        decoded_preds = decode_labels(label_to_index, preds)

        save_predictions(os.path.join(args.predictions, 'preds.{}.txt'.format(dataset)), samples, decoded_truth, decoded_preds, scores)

        if path_scores:
            save_path_scores(os.path.join(args.predictions, 'path_scores.{}.txt'.format(dataset)), path_scores)

        f1 = f1_score(decoded_truth, decoded_preds) * 100
        dataset_loss /= dataset_tokens

        print("[LOG]")
        print("[LOG] {} Loss: {:.4f} F1: {:.3f}".format(dataset.title(), dataset_loss, f1))
        print(classification_report(decoded_truth, decoded_preds, digits=5))
        


def train(model, dataloaders, optimizer, scheduler, label_to_index, args):
    best_f1, best_step = 0., 0
    global_step = 0

    stats = {'train': {'loss': [], 'f1': []}, 'dev': {'loss': [], 'f1': []}}

    epoch_desc = "Epochs (Dev F1: {:.4f} at step {})"
    epoch_iterator = trange(int(args.training.epochs), desc=epoch_desc.format(best_f1, best_step))
    
    for _ in epoch_iterator:
        epoch_iterator.set_description(epoch_desc.format(best_f1, best_step), refresh=True)

        for dataset in ['train', 'dev']:
            if dataset == 'train':
                model.train()
                model.zero_grad()
            else:
                model.eval()

            epoch_tokens = 0
            epoch_loss = 0
            preds, truth = [], []

            batch_iterator = tqdm(dataloaders[dataset], desc=f"{dataset.title()} iteration")
            # ========================================================================
            for batch_i, batch_dict in enumerate(batch_iterator):
                tokens, targets, trends = batch_dict['tokens'], batch_dict['targets'], batch_dict['trends']
                
                result = model(tokens, targets, trends)

                loss = result['loss']

                if dataset == 'train':
                    loss.backward()

                    # Clipping the norm ||g|| of gradient g before the optmizer's step
                    if args.training.clip_grad > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), args.training.clip_grad)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    model.zero_grad()
                    global_step += 1

                # The loss is the mean already
                epoch_tokens += torch.sum(result['mask']).item()
                epoch_loss += loss.item() * torch.sum(result['mask']).item()
                batch_preds = result['tags']

                preds += batch_preds
                truth += [t[:len(tokens[i])] for i, t in enumerate(targets.data.cpu().tolist())]

            # ========================================================================
            decoded_truth = decode_labels(label_to_index, truth)
            decoded_preds = decode_labels(label_to_index, preds)

            epoch_f1 = f1_score(decoded_truth, decoded_preds)
            epoch_loss /= epoch_tokens

            if dataset == 'dev':
                best_f1, best_step = track_best_model(args.checkpoints, model, best_f1, best_step, epoch_f1, epoch_loss, global_step)

            stats[dataset]['loss'].append(epoch_loss)
            stats[dataset]['f1'].append(epoch_f1)

            torch.cuda.empty_cache()
        
    print("[LOG] Done training!")

    print_stats(stats)

    state = torch.load(os.path.join(args.checkpoints, 'model.pt'))
    model.load_state_dict(state['model'])

    print('[LOG] Returning model from step {} with loss {:.4f} and F1 {:.4f}'.format(state['global_step'], state['loss'], state['f1']))
    return model

