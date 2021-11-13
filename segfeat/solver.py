import json
from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from loguru import logger
from pytorch_lightning import LightningModule
from torch import optim
from torch.utils.data import DataLoader

from .dataloader import (BuckeyeDataset, TimitDataset, collate_fn_padd,
                        phoneme_lebels_to_frame_labels,
                        segmentation_to_binary_mask)
from .model import Segmentor
from .utils import PrecisionRecallMetricMultiple, StatsMeter


class Solver(LightningModule):
    def __init__(self, config):
        super(Solver, self).__init__()
        self.hparams = config

        if config.dataset == "timit":
            self.datasetClass = TimitDataset
        elif config.dataset == "buckeye":
            self.datasetClass = BuckeyeDataset
        else:
            raise Exception("invalid dataset type!")
        self.train_dataset, self.valid_dataset, self.test_dataset = self.datasetClass.get_datasets(config)

        self.config = config
        config.rnn_input_size = {'mfcc':  config.n_mfcc * (3 if config.delta_feats else 1) + (4 if config.dist_feats else 0),
                                 'mel':   config.n_mels,
                                 'spect': config.n_fft / 2 + 1}[config.feats]
        config.n_classes = {'timit': 39,
                            'buckeye': 40}[config.dataset]
        self.pr = {'train': PrecisionRecallMetricMultiple(),
                   'val':   PrecisionRecallMetricMultiple(),
                   'test':  PrecisionRecallMetricMultiple()}
        self.phn_acc = {'train': StatsMeter(),
                        'val':   StatsMeter(),
                        'test':  StatsMeter()}
        self.bin_acc = {'train': StatsMeter(),
                        'val':   StatsMeter(),
                        'test':  StatsMeter()}
        self._device = 'cuda' if config.cuda else 'cpu'

        self.build_model()
        logger.info(f"running on {self._device}")
        logger.info(f"rnn input size: {config.rnn_input_size}")
        logger.info(f"{self.segmentor}")

    @pl.data_loader
    def train_dataloader(self):
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.config.batch_size,
                                       shuffle=True,
                                       collate_fn=collate_fn_padd,
                                       num_workers=6)
        logger.info(f"input shape: {self.train_dataset[0][0].shape}")
        logger.info(f"training set length {len(self.train_dataset)}")
        return self.train_loader

    @pl.data_loader
    def val_dataloader(self):
        self.valid_loader = DataLoader(self.valid_dataset,
                                       batch_size=self.config.batch_size,
                                       shuffle=False,
                                       collate_fn=collate_fn_padd,
                                       num_workers=6)
        logger.info(f"validation set length {len(self.valid_dataset)}")
        return self.valid_loader

    @pl.data_loader
    def test_dataloader(self):
        self.test_loader  = DataLoader(self.test_dataset,
                                       batch_size=self.config.batch_size,
                                       shuffle=False,
                                       collate_fn=collate_fn_padd,
                                       num_workers=6)
        logger.info(f"test set length {len(self.test_dataset)}")
        return self.test_loader

    def build_model(self):
        self.segmentor = Segmentor(self.config)

        if self.config.load_ckpt not in [None, "None"]:
            logger.info(f"loading checkpoint from: {self.config.load_ckpt}")
            model_dict = self.segmentor.state_dict()
            weights = torch.load(self.config.load_ckpt, map_location='cuda:0')["state_dict"]
            weights = {k.replace("segmentor.", ""): v for k,v in weights.items()}
            weights = {k: v for k,v in weights.items() if k in model_dict and model_dict[k].shape == weights[k].shape}
            model_dict.update(weights)
            self.segmentor.load_state_dict(model_dict)
            logger.info("loaded checkpoint!")
            if len(weights) != len(model_dict):
                logger.warning("some weights were ignored due to mismatch")
                logger.warning(f"loaded {len(weights)}/{len(model_dict)} modules")
        else:
            logger.info("training from scratch")

    def cls_loss(self, seg, phn_gt, phn_hat):
        """cls_loss
        convert phn_gt to framewise ground truth and take the
        corss-entropy between prediction and truth.

        :param seg: segmentation
        :param phn_gt: list of phonemes for segmentation above
        :param phn_hat: framewise prediction of phonemes
        """
        loss, acc = 0, 0

        for i, (seg_i, phn_gt_i, phn_hat_i) in enumerate(zip(seg, phn_gt, phn_hat)):
            phn_gt_framewise  = phoneme_lebels_to_frame_labels(seg_i, phn_gt_i).to(phn_hat.device)
            phn_hat_i         = phn_hat_i[:len(phn_gt_framewise)]
            loss             += F.cross_entropy(phn_hat_i, phn_gt_framewise)
            acc              += (phn_hat_i.argmax(1) == phn_gt_framewise).sum().item() / len(phn_gt_framewise) * 100

        return loss, acc / len(phn_gt)

    def bin_loss(self, seg, bin_hat):
        """bin_loss
        transform the segmentation to a binary vector with 1 at boundaries
        and 0 elsewhere. take the cross-entropy between precition and truth

        :param seg: segmentation
        :param bin_hat: binary vector, 1s where a boundary is predicted
                        and 0 elsewhere
        """
        loss, acc = 0, 0

        for seg_i, bin_hat_i in zip(seg, bin_hat):
            bin_gt_i   = segmentation_to_binary_mask(seg_i).to(bin_hat.device)
            bin_hat_i  = bin_hat_i[:len(bin_gt_i)]
            loss      += F.cross_entropy(bin_hat_i, bin_gt_i, weight=torch.FloatTensor([0.2, 0.8]).to(bin_hat.device))
            acc       += (bin_hat_i.argmax(1) == bin_gt_i).sum().item() / len(bin_gt_i) * 100

        return loss, acc / len(bin_hat)

    def forward(self, x):
        pass

    def training_step(self, data_batch, batch_i):
        """training_step
        forward 1 training step. calc ranking, phoneme classification
        and boundary classification losses.

        :param data_batch:
        :param batch_i:
        """
        # forward
        spect, seg, phonemes, length = data_batch
        out       = self.segmentor(spect, length, seg)
        loss      = F.relu(1 + out['pred_scores'] - out['gt_scores']).mean()

        phn_loss, phn_acc = self.cls_loss(seg, phonemes, out['cls_out'])
        loss += self.config.phn_cls * phn_loss
        self.phn_acc['train'].update(phn_acc)

        bin_loss, bin_acc = self.bin_loss(seg, out['bin_out'])
        loss += self.config.bin_cls * bin_loss
        self.bin_acc['train'].update(bin_acc)

        # log metrics
        prs = self.pr['train'].update(seg, out['pred'])

        # log into file
        progress = f"[{self.current_epoch}][{batch_i}/{len(self.train_loader)}]"
        for i in range(len(seg)):
            logger.debug("\ny:    {}\nyhat: {}".format(seg[i], out['pred'][i]))
            logger.debug(f"bin output: {out['bin_out'][i].argmax(-1)}")
        logger.info(f"{progress} loss: {loss.item()}")
        logger.info(f"{progress} phn_acc: {phn_acc}, bin_acc: {bin_acc}")
        logger.info(f"{progress} f1: {prs[2][2]}\n")

        return OrderedDict({'loss': loss})

    def generic_eval_step(self, data_batch, batch_i, prefix):
        # forward
        spect, seg, phonemes, length = data_batch
        out       = self.segmentor(spect, length, seg)
        loss      = F.relu(1 + out['pred_scores'] - out['gt_scores']).mean().cpu().item()

        phn_loss, phn_acc = self.cls_loss(seg, phonemes, out['cls_out'])
        loss += self.config.phn_cls * phn_loss.cpu().item()
        self.phn_acc[prefix].update(phn_acc)

        bin_loss, bin_acc = self.bin_loss(seg, out['bin_out'])
        loss += self.config.bin_cls * bin_loss.cpu().item()
        self.bin_acc[prefix].update(bin_acc)

        # log metrics
        self.pr[prefix].update(seg, out['pred'])

        return OrderedDict({f'{prefix}_loss': loss})

    def generic_eval_end(self, outputs, prefix):
        loss_mean = 0

        for output in outputs:
            loss = output[f'{prefix}_loss']
            if self.trainer.use_dp:
                loss = torch.mean(loss)
            loss_mean += loss

        loss_mean /= len(outputs)

        eval_pr       = self.pr[prefix].get_final_metrics()
        train_pr      = self.pr['train'].get_final_metrics()
        eval_phn_acc  = self.phn_acc[prefix].get_stats()
        train_phn_acc = self.phn_acc['train'].get_stats()
        eval_bin_acc  = self.bin_acc[prefix].get_stats()
        train_bin_acc = self.bin_acc['train'].get_stats()
        self.pr[prefix].zero()
        self.pr['train'].zero()
        self.phn_acc[prefix].zero()
        self.phn_acc['train'].zero()
        self.bin_acc[prefix].zero()
        self.bin_acc['train'].zero()

        metrics = OrderedDict({f'avg_{prefix}_loss': loss_mean,
                   f'avg_{prefix}_phn_acc':          eval_phn_acc[0],
                   f'avg_{prefix}_bin_acc':          eval_bin_acc[0],
                   f'std_{prefix}_phn_acc':          eval_phn_acc[1],
                   f'std_{prefix}_bin_acc':          eval_bin_acc[1],
                   f'avg_train_phn_acc':             train_phn_acc[0],
                   f'avg_train_bin_acc':             train_bin_acc[0],
                   f'std_train_phn_acc':             train_phn_acc[1],
                   f'std_train_bin_acc':             train_bin_acc[1]})

        # aggregate val/test metrics
        for level, (precision, recall, f1) in eval_pr.items():
            metrics[f"{prefix}_precision_at_{level}"] = precision
            metrics[f"{prefix}_recall_at_{level}"]    = recall
            metrics[f"{prefix}_f1_at_{level}"]        = f1

        # aggregate train metrics
        for level, (precision, recall, f1) in train_pr.items():
            metrics[f"train_precision_at_{level}"] = precision
            metrics[f"train_recall_at_{level}"]    = recall
            metrics[f"train_f1_at_{level}"]        = f1

        logger.info(f"\nEVAL {prefix} STATS:\n{json.dumps(metrics, sort_keys=True, indent=4)}\n")

        return metrics

    def validation_step(self, data_batch, batch_i):
        return self.generic_eval_step(data_batch, batch_i, 'val')

    def validation_epoch_end(self, outputs):
        return self.generic_eval_end(outputs, 'val')

    def test_step(self, data_batch, batch_i):
        return self.generic_eval_step(data_batch, batch_i, 'test')

    def test_epoch_end(self, outputs):
        return self.generic_eval_end(outputs, 'test')

    def configure_optimizers(self):
        optimizer = {'adam':     torch.optim.Adam(self.segmentor.parameters(), lr=self.config.lr),
                     'sgd':      torch.optim.SGD(self.segmentor.parameters(), lr=self.config.lr, momentum=0.9)}[self.config.optimizer]
        logger.info(f"optimizer: {optimizer}")

        return [optimizer]
