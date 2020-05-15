import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Segmentor(nn.Module):
    def __init__(self, hparams):
        super(Segmentor, self).__init__()
        self.hparams = hparams
        self.device = 'cuda' if hparams.cuda else 'cpu'
        self.min_seg_size = hparams.min_seg_size
        self.max_seg_size = hparams.max_seg_size

        self.rnn = nn.LSTM(hparams.rnn_input_size,
                           hparams.rnn_hidden_size,
                           num_layers=hparams.rnn_layers,
                           batch_first=True,
                           dropout=hparams.rnn_dropout,
                           bidirectional=hparams.birnn)

        # score calculation modules
        self.scorer = nn.Sequential(
                nn.PReLU(),
                nn.Linear((2 if hparams.birnn else 1) * 3 * hparams.rnn_hidden_size, 100),
                nn.PReLU(),
                nn.Linear(100, 1),
                )

        self.classifier = nn.Sequential(
                nn.PReLU(),
                nn.Linear((2 if hparams.birnn else 1) * hparams.rnn_hidden_size, hparams.n_classes * 2),
                nn.PReLU(),
                nn.Linear(hparams.n_classes * 2, hparams.n_classes),
                )

        self.bin_classifier = nn.Sequential(
                nn.PReLU(),
                nn.Linear((2 if hparams.birnn else 1) * hparams.rnn_hidden_size, hparams.n_classes * 2),
                nn.PReLU(),
                nn.Linear(hparams.n_classes * 2, 2),
                )

    def calc_phi(self, rnn_out, rnn_cum):
        batch_size, seq_len, feat_dim = rnn_out.shape

        a = rnn_cum.repeat(1, seq_len, 1)
        b = rnn_cum.repeat(1, 1, seq_len).view(batch_size, -1, feat_dim)
        c = a.sub_(b).view(batch_size, seq_len, seq_len, feat_dim)

        d = rnn_out.repeat(1, 1, seq_len).view(batch_size, seq_len, seq_len, feat_dim)
        e = rnn_out.repeat(1, seq_len, 1).view(batch_size, seq_len, seq_len, feat_dim)
        phi = torch.cat([c, d, e], dim=-1)

        return phi

    def calc_all_scores(self, phi):
        scores = self.scorer(phi).squeeze(-1)
        return scores

    def get_segmentation_score(self, scores, segmentations):
        """get_segmentation_score
        calculate the overall score for a whole segmentation for a batch of
        segmentations

        :param unary_scores:
        :param binary_scores:
        :param segmentations:

        returns: tensor of shape Bx1 where scores[i] = score for segmentation i
        """
        out_scores = torch.zeros((scores.shape[0])).to(scores.device)
        for seg_idx, seg in enumerate(segmentations):
            score = 0
            seg = zip(seg[:-1], seg[1:])
            for start, end in seg:
                score += scores[seg_idx, start, end]
            out_scores[seg_idx] = score

        return out_scores

    def segment_search(self, scores, lengths):
        '''
        Apply dynamic programming algorithm for finding the best segmentation when
        k (the number of segments) is unknown.
        Parameters:
            batch :     A 3D torch tensor: (batch_size, sequence_size, input_size)
            lengths:    A 1D tensor containing the lengths of the batch sequences
            [gold_seg]: A python list containing batch_size lists with the gold
                        segmentations. If given, we will return the best segmentation
                        excluding the gold one, for the structural hinge loss with
                        margin algorithm (see Kiperwasser, Eliyahu, and Yoav Goldberg
                        "Simple and accurate dependency parsing using bidirectional LSTM feature representations).
        Notes:
            The algorithm complexity is O(n**2)
        '''
        batch_size, max_length = scores.shape[:2]
        scores, lengths = scores.to('cpu'), lengths.to('cpu')

        # Dynamic programming algorithm for inference (with batching)
        best_scores = torch.zeros(batch_size, max_length)
        segmentations = [[[0]] for _ in range(batch_size)]

        for i in range(1, max_length):
            # Get scores of subsequences of seq[:i] that ends with i
            start_index = max(0, i - self.max_seg_size)
            end_index = i
            current_scores = torch.zeros(batch_size, end_index - start_index)

            for j in range(start_index, end_index):
                current_scores[:, j - start_index] = best_scores[:, j] + scores[:, j, i]

            # Choose the best scores and their corresponding indexes
            max_scores, k = torch.max(current_scores, 1)
            k = start_index + k # Convert indexes to numpy (relative to the starting index)

            # Add current best score and best segmentation
            best_scores[:, i] = max_scores
            for batch_index in range(batch_size):
                currrent_segmentation = segmentations[batch_index][k[batch_index]] + [i]
                segmentations[batch_index].append(currrent_segmentation)

        # Get real segmentations according to the real lengths of the sequences
        # in the batch
        pred_seg = []
        for i, seg in enumerate(segmentations):
            last_index = lengths[i].item() - 1
            pred_seg.append(seg[last_index])

        return pred_seg

    def forward(self, x, length, gt_seg=None):
        """forward

        :param x:
        :param length:
        """
        results = {}

        # feed through rnn
        x = nn.utils.rnn.pack_padded_sequence(x, length, batch_first=True, enforce_sorted=False)
        rnn_out, _ = self.rnn(x)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        rnn_cum = torch.cumsum(rnn_out, dim=1)
        phi = self.calc_phi(rnn_out, rnn_cum)

        # feed through classifiers
        results['cls_out'] = self.classifier(rnn_out)
        results['bin_out'] = self.bin_classifier(rnn_out)

        # feed through search
        scores = self.calc_all_scores(phi)
        results['pred'] = self.segment_search(scores, length)
        results['pred_scores'] = self.get_segmentation_score(scores, results['pred'])

        if gt_seg is not None:
            results['gt_scores'] = self.get_segmentation_score(scores, gt_seg)

        return results
