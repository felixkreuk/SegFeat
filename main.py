import argparse
import random
import time
from os import mkdir
from os.path import exists, join

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.logging import TestTubeLogger
from torch.backends import cudnn
from torch.utils.data import DataLoader, Dataset

from solver import Solver


def main(hparams):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.benchmark = True

    hparams.run_dir = join(hparams.run_dir, hparams.exp_name)
    logger.info(f"run dir: {hparams.run_dir}")

    log_save_path = join(hparams.run_dir, "run.log")
    logger.add(log_save_path, rotation="500 MB", compression='zip')
    logger.info(f"saving log in: {log_save_path}")

    model_save_path = join(hparams.run_dir, "ckpt")
    logger.info(f"saving models in: {model_save_path}")
    logger.info(f"early stopping with patience of {hparams.patience}")

    solver = Solver(hparams)

    early_stop = EarlyStopping(
        monitor='avg_val_loss',
        patience=hparams.patience,
        verbose=True,
        mode='min'
    )

    tt_logger = TestTubeLogger(
        save_dir=hparams.run_dir,
        name="lightning_logs",
        debug=False,
        create_git_tag=False
    )

    checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        save_top_k=1,
        verbose=True,
        monitor='val_f1_at_2',
        mode='max'
    )

    trainer = Trainer(
            logger=tt_logger,
            overfit_pct=hparams.overfit,
            check_val_every_n_epoch=1,
            min_epochs=1,
            max_epochs=hparams.epochs,
            nb_sanity_val_steps=4,
            checkpoint_callback=None,
            val_percent_check=hparams.val_percent_check,
            val_check_interval=hparams.val_check_interval,
            early_stop_callback=None,
            gpus=hparams.gpus,
            show_progress_bar=False,
            distributed_backend=None,
            )

    if not hparams.test:
        trainer.fit(solver)
    trainer.test(solver)

def parse_args():
    parser = argparse.ArgumentParser(description='segmentation')
    parser.add_argument('--wav_path', type=str)
    parser.add_argument('--dataset', type=str, default='timit', choices=['timit', 'buckeye'])

    parser.add_argument('--run_dir', type=str, default='/tmp/segmentation', help='directory for saving run outputs (logs, ckpt, etc.)')
    parser.add_argument('--exp_name', type=str, default='segmentation_experiment', help='experiment name')
    parser.add_argument('--load_ckpt', type=str, default=None, help='path to a pre-trained model, if provided, training will resume from that point')
    parser.add_argument('--gpus', type=str, default='-1')
    parser.add_argument('--devrun', default=False, action='store_true', help='dev run on a dataset of size `devrun_size`')
    parser.add_argument('--devrun_size', type=int, default=10, help='size of dataset for dev run')
    parser.add_argument('--test', default=False, action='store_true', help='flag to indicate to run a test epoch only (not training will take place)')

    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--epochs', type=int, default=50, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=2, metavar='N', help='batch size')
    parser.add_argument('--dropout', type=float,  default=0.0, help='dropout probability value')
    parser.add_argument('--seed', type=int, default=1245, help='random seed')
    parser.add_argument('--patience', type=int, default=5, help='patience for early stopping')
    parser.add_argument('--gamma', type=float, default=0.15, help='gamma margin')
    parser.add_argument('--overfit', type=int, default=-1, help='gamma margin')
    parser.add_argument('--val_percent_check', type=float, default=1.0, help='how much of the validation set to check')
    parser.add_argument('--val_check_interval', type=float, default=1.0, help='validation check every K epochs')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='precentage of validation from train')

    parser.add_argument('--rnn_input_size', type=int, default=50, help='number of inputs')
    parser.add_argument('--rnn_hidden_size', type=int, default=200, help='RNN hidden layer size')
    parser.add_argument('--rnn_dropout', type=float, default=0.3, help='dropout')
    parser.add_argument('--birnn', default=True, help='BILSTM, if define will be biLSTM')
    parser.add_argument('--rnn_layers', type=int, default=2, help='number of lstm layers')
    parser.add_argument('--min_seg_size', type=int, default=1, help='minimal size of segment, examples with segments smaller than this will be ignored')
    parser.add_argument('--max_seg_size', type=int, default=100, help='see `min_seg_size`')
    parser.add_argument('--max_len', type=int, default=500, help='maximal size of sequences')
    parser.add_argument('--feats', type=str, default="mfcc", choices=["mfcc", "mel", "spect"], help='type of acoustic features to use')
    parser.add_argument('--random_trim', default=False, action="store_true", help='if this flag is on seuqences will be randomly trimmed')
    parser.add_argument('--delta_feats', default=False, action="store_true", help='if this flag is on delta features will be added')
    parser.add_argument('--dist_feats', default=False, action="store_true", help='if this flag is on the euclidean features will be added (see paper)')
    parser.add_argument('--normalize', default=False, action="store_true", help='flag to normalize features')
    parser.add_argument('--bin_cls', default=0, type=float, help='coefficient of binary classification loss')
    parser.add_argument('--phn_cls', default=0, type=float, help='coefficient of phoneme classification loss')
    parser.add_argument('--n_fft', type=int, default=160, help='n_fft for feature extraction')
    parser.add_argument('--hop_length', type=int, default=160, help='hop_length for feature extraction')
    parser.add_argument('--n_mels', type=int, default=40, help='number of mels')
    parser.add_argument('--n_mfcc', type=int, default=13, help='number of mfccs')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.cuda = torch.cuda.is_available()
    main(args)
