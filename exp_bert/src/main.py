import os
import re
import json
import torch
import argparse
import exp_bert.src.data.dataset as ds
import exp_bert.src.modeling.nets as nets
import exp_bert.src.commons.utilities as utils

from tqdm import tqdm, trange
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoConfig, AutoModel, AdamW, get_linear_schedule_with_warmup

from fairseq.data import Dictionary
from fairseq.data.encoders.fastbpe import fastBPE


class Arguments(dict):
    def __init__(self, *args, **kwargs):
        super(Arguments, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @staticmethod
    def from_nested_dict(data):
        if not isinstance(data, dict):
              return data
        else: return Arguments({key: Arguments.from_nested_dict(data[key]) for key in data})


def get_dataloaders(args, tokenizer, bpe, vocab):
    dargs = args.data
    pargs = args.preproc
    oargs = args.optim

    corpus = dict()
    for split in dargs.partitions:
        if split == 'train' or split == 'dev':
            splits, fnames = [split], [dargs.partitions[split]]
        else:
            fnames = dargs.partitions[split]
            splits = [os.path.splitext(fname)[0] for fname in fnames]

        if split == 'train':
            colnames = dargs.colnames
        else:
            colnames = {'tokens': 0, 'labels': 1}
            
        for split, fname in zip(splits, fnames):
            fpath = os.path.join(dargs.directory, fname)

            if pargs.dataset_class == 'bert':
                dataset = ds.NERDataset(
                    fpath, colnames, dargs.label_scheme, tokenizer, bpe, vocab, split, use_tokenizer=True
                )
            elif pargs.dataset_class == 'roberta':
                dataset = ds.NERDataset(
                    fpath, colnames, dargs.label_scheme, tokenizer, bpe, vocab, split, use_tokenizer=False
                )
            else:
                raise NotImplementedError("Unexpected dataset class")

            if split == 'train':
                oargs.train_batch_size = oargs.per_gpu_train_batch_size * max(1, oargs.n_gpu)
                batch_size = oargs.train_batch_size
                sampler = RandomSampler(dataset)
            else:
                oargs.eval_batch_size = oargs.per_gpu_eval_batch_size * max(1, oargs.n_gpu)
                batch_size = oargs.eval_batch_size
                sampler = SequentialSampler(dataset)
            corpus[split] = DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=dataset.collate)
    return corpus


def get_model_class(model_name):
    if model_name == 'NERModel':
        model_class = nets.NERModel

    else:
        raise NotImplementedError(f'Unknown model name: {model_name}')

    return model_class


def prepare_model_config(args):
    if args.model.name == 'NERModel':
        config = AutoConfig.from_pretrained(args.model.config)
        config.pretrained_frozen = args.model.pretrained_frozen
        config.model_name_or_path = args.model.model_name_or_path
        config.num_labels = len(args.data.label_scheme)
        config.output_hidden_states = False
        config.output_attentions = False

    else:
        raise NotImplementedError(f"Unexpected model name: {args.model.name}")

    return config


def get_optimizer(args, model):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate,
                      betas=(args.beta_1, args.beta_2),
                      eps=args.adam_epsilon)
    return optimizer


def track_best_model(args, model, dev_stats, best_f1, best_step, global_step):
    curr_f1, _, _ = dev_stats.metrics(args.data.label_scheme)
    if best_f1 > curr_f1:
        return best_f1, best_step

    # Save model checkpoint
    os.makedirs(args.experiment.checkpoint_dir, exist_ok=True)
    model_to_save = (model.module if hasattr(model, "module") else model)  # Take care of distributed/parallel training
    model_to_save.save_pretrained(args.experiment.checkpoint_dir)
    meta = {
        'args': args,
        'f1': curr_f1,
        'global_step': global_step
    }
    torch.save(meta, os.path.join(args.experiment.checkpoint_dir, "training_meta.bin"))
    return curr_f1, global_step


def print_stats(stats, label_scheme):
    for i in range(len(stats['train'])):
        print(f"Epoch {i + 1} -", end=" ")
        for split in ['train', 'dev']:
            epoch_stats = stats[split][i]
            f1, _, _ = epoch_stats.metrics(label_scheme)
            loss = sum(epoch_stats.losses) / len(epoch_stats.losses)
            print(f"[{split.upper()}] F1: {f1 * 100:.3f} Loss: {loss:.5f}", end=' ')
        print()
    print()


def train(args, model, dataloaders):
    oargs = args.optim

    if oargs.max_steps > 0:
        t_total = oargs.max_steps
        oargs.num_train_epochs = oargs.max_steps // (len(dataloaders['train']) // oargs.gradient_accumulation_steps) + 1
    else:
        t_total = len(dataloaders['train']) // oargs.gradient_accumulation_steps * oargs.num_train_epochs

    optimizer = get_optimizer(oargs, model)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=oargs.warmup_steps, num_training_steps=t_total)

    # multi-gpu training
    if oargs.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    model.zero_grad()
    utils.set_seed(args.experiment.seed)  # Added here for reproductibility

    best_f1, best_step = 0., 0
    global_step = 0
    stats = {'train': [], 'dev': []}

    epoch_desc = "Epochs (Dev F1: {:.5f} at step {})"
    epoch_iterator = trange(int(args.optim.num_train_epochs), desc=epoch_desc.format(best_f1, best_step))

    for _ in epoch_iterator:
        epoch_iterator.set_description(epoch_desc.format(best_f1, best_step), refresh=True)

        for split in ['train', 'dev']:
            epoch_stats = utils.EpochStats()
            batch_iterator = tqdm(dataloaders[split], desc=f"{split.title()} iteration")
            # ====================================================================
            for step, batch in enumerate(batch_iterator):
                if split == 'train':
                    model.train()
                    model.zero_grad()
                else:
                    model.eval()

                for field in batch.keys():
                    if batch[field] is not None:
                        batch[field] = batch[field].to(oargs.device)

                outputs = model(**batch, wrap_scalars=oargs.n_gpu > 1)
                loss = outputs[0]

                if oargs.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training

                if oargs.gradient_accumulation_steps > 1:
                    loss = loss / oargs.gradient_accumulation_steps

                epoch_stats.step(scores=outputs[1], target=batch['labels'], mask=batch['label_mask'], loss=loss.item())

                if split == 'train':
                    loss.backward()

                    if (step + 1) % oargs.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), oargs.max_grad_norm)

                        optimizer.step()
                        scheduler.step()  # Update learning rate schedule
                        model.zero_grad()
                        global_step += 1

                if oargs.max_steps > 0 and global_step > oargs.max_steps:
                    batch_iterator.close()
                    break
            # ====================================================================
            stats[split].append(epoch_stats)

            if split == 'dev':
                best_f1, best_step = track_best_model(args, model, epoch_stats, best_f1, best_step, global_step)
        
        os.makedirs(args.experiment.output_dir, exist_ok=True)
        torch.save(stats['train'], os.path.join(args.experiment.output_dir, 'train_preds_across_epochs.bin'))
        torch.save(stats['dev'], os.path.join(args.experiment.output_dir, 'dev_preds_across_epochs.bin'))

        if oargs.max_steps > 0 and global_step > oargs.max_steps:
            epoch_iterator.close()
            break
    return stats, best_f1, best_step


def predict(args, model, dataloader):
    model.eval()
    stats = utils.EpochStats()

    oargs = args.optim

    # multi-gpu evaluate
    if oargs.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    for batch in tqdm(dataloader):
        for field in batch:
            if batch[field] is not None:
                batch[field] = batch[field].to(oargs.device)

        outs = model(**batch, wrap_scalars=oargs.n_gpu > 1)
        loss = outs[0]

        if oargs.n_gpu > 1:
            # There is one parallel loss per device
            loss = loss.mean()

        stats.step(scores=outs[1], target=batch['labels'], mask=batch['label_mask'], loss=loss.item())
        
    return stats


def load_args(default_config=None, verbose=False):
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--config', default=default_config, type=str, required=default_config is None, help='Provide the JSON config file with the experiment parameters')
    
    parser.add_argument('--mode', choices=['train', 'eval'], default='train')

    if default_config is None:
        arguments = parser.parse_args()
    else:
        arguments = parser.parse_args("")

    # Override the default values with the JSON arguments
    with open(os.path.join(utils.PROJ_DIR, arguments.config)) as f:
        params = ''.join([re.sub(r"//.*$", "", line, flags=re.M) for line in f])  # Remove comments from the JSON config
        args = Arguments.from_nested_dict(json.loads(params))

    # Training Mode ['train', 'eval']
    args.mode = arguments.mode

    # Data Args
    args.data.directory = os.path.join(utils.PROJ_DIR, args.data.directory)

    # Exp Args
    args.experiment.output_dir = os.path.join(utils.PROJ_DIR, args.experiment.output_dir, args.experiment.id)
    args.experiment.checkpoint_dir = os.path.join(args.experiment.output_dir, "checkpoint")

    # Optim Args
    args.optim.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.optim.n_gpu = torch.cuda.device_count()

    if verbose:
        for main_field in ['experiment', 'data', 'preproc', 'model', 'optim']:
            assert hasattr(args, main_field)
            print(f"{main_field.title()} Args:")
            for k,v in args[main_field].items():
                print(f"\t{k}: {v}")
            print()
    return args


def load_bpe_and_vocab(args):
    if args.model.model_class == 'roberta':
        args.model.bpe_codes = os.path.join(utils.PROJ_DIR, args.model.bpe_codes)
        print(f"Loading BPE from pretrained checkpoint at {args.model.bpe_codes}")
        bpe = fastBPE(args.model)

        args.model.vocab = os.path.join(utils.PROJ_DIR, args.model.vocab)
        print(f"Loading BPE from pretrained checkpoint at {args.model.vocab}")
        vocab = Dictionary()
        vocab.add_from_file(args.model.vocab)
        print()
    else:
        bpe, vocab = None, None

    return bpe, vocab


def load_checkpoint(args, model, ckpt_path):

    if args.model.model_class == 'bert':
        model.bert = AutoModel.from_pretrained(ckpt_path)
        return model
    elif args.model.model_class == 'roberta':
        print()
        print('Loading weights...')
        old_state = model.state_dict()
        new_state = torch.load(ckpt_path, map_location=args.optim.device)
        for pname in old_state.keys():
            pname_norm = pname.replace('bert.', 'roberta.')
            if pname_norm in new_state:
                old_state[pname] = new_state[pname_norm]
            else:
                print("[LOG] Missing: {} ({})".format(pname, pname_norm))
        model.load_state_dict(old_state)
    
    print()
    return model


def main():

    args = load_args()

    print("GPUs available:", args.optim.n_gpu)
    utils.set_seed(args.experiment.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model.tokenizer, do_lower_case=args.preproc.do_lowercase)

    print(f"Vocab size: {len(tokenizer)}")
    if len(args.preproc.new_tokens) > 0:
        tokenizer.add_tokens(args.preproc.new_tokens)
        print(f"Adding new tokens. New vocab size: {len(tokenizer)}")

    bpe, vocab = load_bpe_and_vocab(args)

    dataloaders = get_dataloaders(args, tokenizer, bpe, vocab)

    print(f"Reading dataset from '{args.data.directory.replace(utils.PROJ_DIR, '$PROJECT')}'")
    for split in dataloaders:
        print(f"{split.upper()}: {len(dataloaders[split].dataset)}")
    print()

    config = prepare_model_config(args)
    model = get_model_class(args.model.name)(config)
    model = load_checkpoint(args, model, args.model.model_name_or_path)
    model.to(args.optim.device)

    if len(args.preproc.new_tokens) > 0:
        model.resize_token_embeddings(len(tokenizer))
        assert len(tokenizer) == model.config.vocab_size
        assert len(tokenizer) == config.vocab_size

    if args.mode == 'train':
        confirm = '0'
        if os.path.exists(args.experiment.checkpoint_dir):
            confirm = input("[LOG] Found a checkpoint! Choose an option:\n"
                           "\t0) Train from scratch and override the previous checkpoint\n"
                           "\t1) Load the checkpoint and train from there\nYour choice: ")

        if confirm == '1':
            print(f"Loading model from pretrained checkpoint at {args.experiment.checkpoint_dir}")
            model = model.from_pretrained(args.experiment.checkpoint_dir)
            model.to(args.optim.device)

        # with torch.autograd.detect_anomaly():
        stats, f1, global_step = train(args, model, dataloaders)
        print_stats(stats, args.data.label_scheme)

        print(f"\nBest dev F1: {f1:.5f}")
        print(f"Best global step: {global_step}")
    else:
        print("Skipping training")
        stats = {
            'train': torch.load(os.path.join(args.experiment.output_dir, 'train_preds_across_epochs.bin')),
            'dev': torch.load(os.path.join(args.experiment.output_dir, 'dev_preds_across_epochs.bin'))
        }
        print_stats(stats, args.data.label_scheme)

    # Load the best checkpoint according to dev
    if os.path.exists(args.experiment.checkpoint_dir):
        print(f"Loading model from pretrained checkpoint at {args.experiment.checkpoint_dir}")
        model = model.from_pretrained(args.experiment.checkpoint_dir)
        model.to(args.optim.device)

    if utils.input_with_timeout("Do you want to evaluate the model? [y/n]:", 15, "y").strip() == 'y':
        # Perform evaluation over the dev and test sets with the best checkpoint
        for split in dataloaders.keys():
            if split == 'train':
                continue

            stats = predict(args, model, dataloaders[split])
            torch.save(stats, os.path.join(args.experiment.output_dir, f'{split}_best_preds.bin'))

            loss, _, _ = stats.loss()

            report = stats.get_classification_report(args.data.label_scheme)
            classes = sorted(set([label[2:] for label in args.data.label_scheme if label != 'O']))

            print(f"\n********** {split.upper()} RESULTS **********\n")
            print('\t'.join(["Loss"] + classes + ["F1"]), end='\n')
            print('\t'.join([f"{l:.4f}" for l in [loss]]), end='\t')
            f1_scores = []
            for c in classes + ["micro avg"]:
                if 'f1-score' in report[c].keys():
                    f1_scores.append(report[c]['f1-score'])
                else:
                    f1_scores.append(0)
            print('\t'.join([f"{score * 100:.3f}" for score in f1_scores]))
            print()

            if utils.input_with_timeout("Print class-level results? [y/n]:", 5, "n").strip() == 'y':
                stats.print_classification_report(report=report)
        print()


if __name__ == '__main__':
    main()

