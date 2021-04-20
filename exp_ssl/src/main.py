import os
import re
import json
import torch
import random
import argparse
import numpy as np
import exp_ssl.src.commons.globals as glb
import exp_ssl.src.modeling.experiment as exp

from types import SimpleNamespace as Namespace


class Arguments(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', default='configs/exp1/exp1.0.wnut.json', help='Provide the JSON config path with the parameters of your experiment')
        parser.add_argument('--mode', choices=['train', 'eval'], default='train')
        parser.add_argument('--replicable', action='store_true', help='If provided, a seed will be used to allow replicability')

        args = parser.parse_args()

        # Fields expected from the command line
        self.config = os.path.join(glb.PROJ_DIR, args.config)
        self.mode = args.mode
        self.replicable = args.replicable

        assert os.path.exists(self.config) and self.config.endswith('.json'), 'The config path provided does not exist or is not a JSON file'

        # Read the parameters from the JSON file and skip comments
        with open(self.config, 'r') as f:
            params = ''.join([re.sub(r"//.*$", "", line, flags=re.M) for line in f])

        arguments = json.loads(params, object_hook=lambda d: Namespace(**d))

        # Must-have fields expected from the JSON config file
        self.experiment_id = arguments.experiment_id
        self.description = arguments.description
        self.dataset = arguments.dataset
        self.model = arguments.model
        self.training = arguments.training
        self.evaluation = arguments.evaluation

        self.training.optim.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training.optim.n_gpu = torch.cuda.device_count()

        # Checking that the JSON contains at least the fixed fields
        assert all([hasattr(self.dataset, name) for name in {'train', 'dev', 'test'}])
        assert all([hasattr(self.model, name) for name in {'name'}])
        assert all([hasattr(self.training, name) for name in {'epochs', 'batch_size', 'optim', 'lr_scheduler', 'clip_grad'}])
        assert all([hasattr(self.training.optim, name) for name in {'name', 'lr', 'weight_decay'}])
        assert all([hasattr(self.evaluation, name) for name in {'batch_size'}])

        self._format_datapaths()
        self._add_extra_fields()

    def _format_datapaths(self):
        self.dataset.train = os.path.join(glb.DATA_DIR, self.dataset.train)
        self.dataset.dev = os.path.join(glb.DATA_DIR, self.dataset.dev)
        self.dataset.test = os.path.join(glb.DATA_DIR, self.dataset.test)

    def _add_extra_fields(self):
        self.checkpoints = os.path.join(glb.CHECKPOINT_DIR, self.experiment_id)
        self.figures = os.path.join(glb.FIGURE_DIR, self.experiment_id)
        self.history = os.path.join(glb.HISTORY_DIR, self.experiment_id)
        self.predictions = os.path.join(glb.PREDICTIONS_DIR, self.experiment_id)


def main():
    args = Arguments()

    if args.replicable:
        seed_num = 123
        random.seed(seed_num)
        np.random.seed(seed_num)
        torch.manual_seed(seed_num)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_num)
            torch.backends.cudnn.deterministic = True


    glb.DEVICE = args.training.optim.device

    print("[LOG] {}".format('=' * 40))
    print("[LOG] {: >15}: '{}'".format("Experiment ID", args.experiment_id))
    print("[LOG] {: >15}: '{}'".format("Description", args.description))
    for key, val in vars(args.dataset).items():
        print("[LOG] {: >15}: {}".format(key, val))
    print("[LOG] {: >15}: '{}'".format("Modeling", vars(args.model)))
    print("[LOG] {: >15}: '{}'".format("Training", vars(args.training)))
    print("[LOG] {: >15}: '{}'".format("Evaluation", vars(args.evaluation)))
    print("[LOG] {: >15}: '{}'".format("Device", glb.DEVICE))
    print("[LOG] {: >15}: '{}'".format("GPUs available", args.training.optim.n_gpu))
    print("[LOG] {}".format('=' * 40))

    exp.main(args)

if __name__ == '__main__':
    main()
