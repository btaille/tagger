import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence

from tqdm import tqdm

from time import time

from torch_utils import prepare_sequence

import numpy as np


def cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor


def clip_grad(v, min, max):
    v.register_hook(lambda g: g.clamp(min, max))
    return v


def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def log_sum_exp_mat(mat):
    max_score_broadcast = torch.max(mat, 0)[0].view(1, -1).repeat(mat.size(1), 1)
    return max_score_broadcast[0] + \
           torch.log(torch.sum(torch.exp(mat - max_score_broadcast), 0))


def log_sum_exp3d(mat):
    batch_size = mat.size(0)
    max_score_broadcast = torch.max(mat, 1)[0].view(batch_size, 1, -1).repeat(1, mat.size(1), 1)

    return max_score_broadcast[:, 0, :] + \
           torch.log(torch.sum(torch.exp(mat - max_score_broadcast), 1)).view(batch_size, -1)


class Tagger(nn.Module):
    def __init__(self, w_embeddings, hidden_dim, idx2tag, w2idx=None, freeze_embeddings=False,
                 dropout=0., crf=False):
        super(Tagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.idx2tag = idx2tag
        self.tag2idx = {v: k for (k, v) in idx2tag.items()}
        self.tagset_size = len(self.idx2tag)

        self.word_embeddings = nn.Embedding(w_embeddings.size(0), w_embeddings.size(1))
        self.word_embeddings.weight = nn.Parameter(w_embeddings)
        self.dropout = dropout
        self.drop = nn.Dropout(self.dropout)

        self.lstm = nn.LSTM(w_embeddings.size(1), hidden_dim // 2, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.hidden = self.init_hidden()

        self.crf = crf

        self.word_index = w2idx

        if freeze_embeddings:
            self.word_embeddings.weight.requires_grad = False

        self.transitions = cuda(nn.Parameter(torch.randn(self.tagset_size, self.tagset_size)))

    def init_hidden(self, batch_size=1):
        return (cuda(autograd.Variable(torch.randn(2, batch_size, self.hidden_dim // 2))),
                cuda(autograd.Variable(torch.randn(2, batch_size, self.hidden_dim // 2))))

    def init_hidden_char(self, batch_size=1):
        return (cuda(autograd.Variable(torch.randn(1, batch_size, self.char_hidden_dim // 2))),
                cuda(autograd.Variable(torch.randn(1, batch_size, self.char_hidden_dim // 2))))

    def init_char_embeddings(self, sent_len, batch_size=1):
        return cuda(autograd.Variable(torch.randn(sent_len, batch_size, self.char_hidden_dim // 2)))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = cuda(torch.Tensor(1, self.tagset_size).fill_(-10000.))
        # START_TAG has all of the score.
        init_alphas[0][self.tag2idx["<START>"]] = 0.

        forward_var = autograd.Variable(init_alphas)
        # Optimized forward pass
        for feat in feats:
            alphas = forward_var.view(-1, 1).repeat(1, self.tagset_size)
            emit_scores = feat.view(1, -1).repeat(self.tagset_size, 1)
            next_tag_vars = alphas + self.transitions.transpose(0, 1) + emit_scores
            forward_var = log_sum_exp_mat(next_tag_vars)

        terminal_var = forward_var.view(1, -1) + self.transitions[self.tag2idx["<STOP>"]]
        alpha = log_sum_exp(terminal_var)

        return alpha

    def _forward_alg_batch(self, feats, lens):
        batch_size = feats.size(0)
        # Do the forward algorithm to compute the partition function
        init_alphas = cuda(torch.Tensor(batch_size, self.tagset_size, 1).fill_(-10000.))
        # START_TAG has all of the score.
        init_alphas[:, self.tag2idx["<START>"], 0] = 0.

        forward_var = autograd.Variable(init_alphas)

        trans = self.transitions.transpose(0, 1).contiguous().view(1, self.tagset_size, self.tagset_size).repeat(
            batch_size, 1, 1)
        stop_trans = self.transitions[self.tag2idx["<STOP>"]].view(1, -1).repeat(batch_size, 1)
        terminal_var = stop_trans

        # Optimized forward pass
        for i, feat in enumerate(feats.transpose(0, 1)):
            alphas = forward_var.view(batch_size, -1, 1).repeat(1, 1, self.tagset_size)
            emit_scores = feat.view(batch_size, 1, -1).repeat(1, self.tagset_size, 1)

            next_tag_vars = alphas + trans + emit_scores
            forward_var = log_sum_exp3d(next_tag_vars)

            if i+1 in set(lens):
                for k in range(len(lens)):
                    if lens[k] == i+1:
                        terminal_var[k] = terminal_var[k] + forward_var[k]

        alpha = log_sum_exp3d(terminal_var)

        return alpha.view(-1)

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = cuda(autograd.Variable(torch.Tensor([0])))

        tags = torch.cat([cuda(torch.LongTensor([self.tag2idx["<START>"]])), tags.data])

        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag2idx["<STOP>"], tags[-1]]
        return score


    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = cuda(torch.Tensor(1, self.tagset_size).fill_(-10000.))
        init_vvars[0][self.tag2idx["<START>"]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = autograd.Variable(init_vvars)
        for feat in feats:
            # Optimized forward pass
            alphas = forward_var.view(-1, 1).repeat(1, self.tagset_size)
            next_tag_vars = alphas + self.transitions.transpose(0, 1)
            viterbivars, bptrs_t = torch.max(next_tag_vars, 0)
            bptrs_t = np.concatenate(bptrs_t.cpu().data.numpy()).tolist()
            forward_var = (viterbivars + feat).view(1, -1)
            backpointers.append(bptrs_t)

            # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag2idx["<STOP>"]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag2idx["<START>"]  # Sanity check
        best_path.reverse()
        return path_score, best_path


    def neg_log_likelihood(self, sentences, lens, tags, test=0, gradient_clipping=0):
        if len(sentences.size()) > 1:
            batch_size = sentences.size(0)
        else:
            batch_size = 1

        self.zero_grad()
        self.hidden = self.init_hidden(batch_size=batch_size)

        w_embeds = self.word_embeddings(sentences)

        # ner
        if test:
            inputs = (1. - self.dropout) * w_embeds
        else:
            inputs = self.drop(w_embeds)

        packed = pack_padded_sequence(inputs, lens.numpy(), batch_first=True)
        lstm_out, self.hidden = self.lstm(packed, self.hidden)

        lstm_feats = self.hidden2tag(lstm_out.data.view(-1, self.hidden_dim))

        padded_feats = pad_packed_sequence(PackedSequence(lstm_feats, lstm_out.batch_sizes), batch_first=True)

        #         forward_score = self._forward_alg(padded_feats)

        if not test and gradient_clipping:
            # lstm_out = clip_grad(lstm_out, - gradient_clipping,  gradient_clipping)
            lstm_feats = clip_grad(lstm_feats, - gradient_clipping, gradient_clipping)
            # forward_score = clip_grad(forward_score, - gradient_clipping,  gradient_clipping)

        gold_scores = cuda(autograd.Variable(torch.zeros((batch_size))))
        lens = padded_feats[1]

        for i, feat in enumerate(padded_feats[0]):
            gold_scores[i] = self._score_sentence(feat[:lens[i]], tags[i][:lens[i]].view(-1))

        forward_scores = self._forward_alg_batch(padded_feats[0], lens)

        return torch.mean(forward_scores - gold_scores)


    def forward(self, sentences, lens, test=0, gradient_clipping=0):
        if len(sentences.size()) > 1:
            batch_size = sentences.size(0)
        else:
            batch_size = 1

        self.zero_grad()
        self.hidden = self.init_hidden(batch_size=batch_size)

        w_embeds = self.word_embeddings(sentences)

        # ner
        if test:
            inputs = (1. - self.dropout) * w_embeds
        else:
            inputs = self.drop(w_embeds)

        packed = pack_padded_sequence(w_embeds, lens.numpy(), batch_first=True)
        lstm_out, self.hidden = self.lstm(packed, self.hidden)

        lstm_feats = self.hidden2tag(lstm_out.data.view((-1, self.hidden_dim)))

        if not test and gradient_clipping:
            lstm_out = clip_grad(lstm_out, - gradient_clipping, gradient_clipping)
            lstm_feats = clip_grad(lstm_feats, - gradient_clipping, gradient_clipping)

        if self.crf:
            scores = []
            tag_seqs = []

            padded_feats = pad_packed_sequence(PackedSequence(lstm_feats, lstm_out.batch_sizes), batch_first=True)

            for i, l in enumerate(lens):
                score, tag_seq = self._viterbi_decode(padded_feats[0][i, :l, :])
                scores.append(score)
                tag_seqs.append(tag_seq)

            if not test and gradient_clipping:
                score = clip_grad(score, - gradient_clipping, gradient_clipping)

            return scores, tag_seqs

        else:
            tag_scores = F.log_softmax(lstm_feats)

            if not test and gradient_clipping:
                tag_scores = clip_grad(tag_scores, - gradient_clipping, gradient_clipping)

            # tag_scores = pad_packed_sequence((tag_scores, batch_out))

            return PackedSequence(tag_scores, lstm_out.batch_sizes)

    def test(self, loader):
        self.volatile = True
        preds = []
        losses = []
        padded_scores = []

        if not self.crf:
            for i, (sentences, tags, lens) in tqdm(enumerate(loader)):
                sentences_in = autograd.Variable(cuda(sentences[:, :lens.numpy()[0]]), volatile=True)
                targets = autograd.Variable(cuda(tags[:, :lens.numpy()[0]]), volatile=True)
                packed_targets = pack_padded_sequence(targets, lens.numpy(), batch_first=True)

                scores = self.forward(sentences_in, lens, test=1)

                padded_scores.append(pad_packed_sequence(scores))
                losses.append(nn.CrossEntropyLoss()(scores.data, packed_targets.data).cpu().data.numpy()[0])

            for score, lens in padded_scores:
                pred_batch = np.argmax(score.cpu().data.numpy(), axis=2).transpose()
                for i, p in enumerate(pred_batch):
                    preds.append(pred_batch[i, :lens[i]].tolist())

        else:
            for i, (sentences, tags, lens) in tqdm(enumerate(loader)):
                sentences_in = autograd.Variable(cuda(sentences[:, :lens.numpy()[0]]), volatile=True)
                targets = autograd.Variable(cuda(tags[:, :lens.numpy()[0]]), volatile=True)
                packed_targets = pack_padded_sequence(targets, lens.numpy(), batch_first=True)

                neg_log_likelihood = self.neg_log_likelihood(sentences_in, lens, targets)

                preds.append(self.forward(sentences_in, lens, test=1)[1])

                losses.append(neg_log_likelihood.cpu().data.numpy())

            cat = []
            for l in preds:
                for p in l:
                    cat.append(p)
            preds = cat

        #         for i, (s, t, l) in enumerate(zip(tqdm(sentences), tags)):
        #             inputs = prepare_sequence(s, volatile=True)
        #             targets = prepare_sequence(t, volatile=True)

        #             if self.crf:
        #                 neg_log_likelihood = self.neg_log_likelihood(inputs, targets)

        #                 preds.append(self.forward(prepare_sequence(s), test=1)[1])
        #                 losses.append(neg_log_likelihood.cpu().data.numpy())

        #             else:
        #                 scores = self.forward(inputs, test=1)

        #                 preds.append(scores.cpu().data.numpy())
        #                 losses.append(nn.NLLLoss()(scores, targets).cpu().data.numpy())

        self.volatile = False

        return preds, np.mean(losses)
