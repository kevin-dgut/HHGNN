import argparse

import dgl
import random

import numpy as np
import torch


class parase_opt():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch MHHGT')

        parser.add_argument('--train', default=0, type=int, help='train mode or evaluation mode')
        parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
        parser.add_argument('--wd', default=0.01, type=float, help='weight decay')
        parser.add_argument('--epochs', default=50, type=int, help='number of epochs')
        parser.add_argument('--dropout', default=0.5, type=float, help='dropout rate')
        parser.add_argument('--num_classes', default=2, type=int, help='number of classes')
        parser.add_argument('--n_folds', default=5, type=int, help='number of folds')
        parser.add_argument('--best_model_path', default='./', type=str, help='path to checkpoint file')
        parser.add_argument('--device',  type=str, help='device')
        parser.add_argument('--batch_size', default=40, type=int, help='batch size')
        parser.add_argument('--graph_path',default="./")
        parser.add_argument('--atlas_dir',default="./")
        parser.add_argument('--log_path', default='./', type=str,help='path to log file')
        parser.add_argument('--netmatrix_schaefer_path',default='./', type=str, help='path to path file')
        parser.add_argument('--num_workers', default=4, type=int, help='num_workers')
        parser.add_argument('--task',type=str,default='NC_EMCI',help='Task for classfication')
        parser.add_argument('--k',type=int, default=650,help='Number of k values')

        args = parser.parse_args()



        if args.device.startswith('cuda'):
            device_id = int(args.device.split(':')[1])
            if torch.cuda.is_available() and device_id < torch.cuda.device_count():
                args.device = torch.device(f'cuda:{device_id}')

            else:
                print(f"CUDA device {device_id} is not available, using CPU instead.")
                args.device = torch.device('cpu')
        else:
            args.device = torch.device('cpu')
        print(f"Using device: {args.device}")

        self.args = args

    def print_args(self):
        print('\nParameters:')
        for arg, content in self.args.__dict__.items():
            print('\t{}={}'.format(arg, content))
        print('\n')

        phase = "train" if self.args.train == 1 else "eval"
        print('phase:', phase)

    def initialize(self):
        self.setseed(42)
        self.print_args()
        return self.args

    def setseed(self, seed=0):

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        dgl.seed(seed)
        dgl.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False