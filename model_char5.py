import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence

from tqdm import tqdm

import os
import pickle
import numpy as np

from eval import eval_metrics_crf, save_plot
from loader import loader, order
from torch_utils import cuda

from crf import CRF


class Tagger(nn.Module):
    def __init__(self, model_path, w_embeddings, hidden_dim, w2idx, c2idx, tag2idx, char_embeddings=None,
                 char_hidden_dim=None, freeze_embeddings=False, dropout=0., crf=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.w2idx = w2idx
        self.c2idx = c2idx
        self.tag2idx = tag2idx
        self.idx2tag = {v: k for (k, v) in tag2idx.items()}
        self.tagset_size = len(self.idx2tag) + 2
        self.model_path = model_path

        self.init_metrics()

        self.word_embeddings = nn.Embedding(w_embeddings.size(0), w_embeddings.size(1))
        self.word_embeddings.weight = nn.Parameter(w_embeddings)

        if char_embeddings is not None and char_hidden_dim is not None:
            self.char_dim = char_embeddings.size(1)
            self.char_embeddings = nn.Embedding(char_embeddings.size(0), char_embeddings.size(1))
            self.char_embeddings.weight = nn.Parameter(char_embeddings)
            self.char = True

            # character-level word embeddings
            self.char_hidden_dim = char_hidden_dim
            self.lstm_char = nn.LSTM(char_embeddings.size(1), self.char_hidden_dim // 2, bidirectional=True)
            self.hidden_char = self.init_hidden_char()

        else:
            self.char = False

        self.dropout = dropout
        self.drop = nn.Dropout(self.dropout)

        if self.char:
            self.lstm = nn.LSTM(w_embeddings.size(1) + self.char_hidden_dim, hidden_dim // 2, bidirectional=True)
        else:
            self.lstm = nn.LSTM(w_embeddings.size(1), hidden_dim // 2, bidirectional=True)

        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.hidden = self.init_hidden()

        self.crf = crf
        if self.crf:
            self.CRF = CRF(self.tag2idx)

        self.word_index = w2idx

        if freeze_embeddings:
            self.word_embeddings.weight.requires_grad = False

    def init_hidden(self, batch_size=1):
        return (cuda(autograd.Variable(torch.randn(2, batch_size, self.hidden_dim // 2))),
                cuda(autograd.Variable(torch.randn(2, batch_size, self.hidden_dim // 2))))

    def init_hidden_char(self, batch_size=1):
        return (cuda(autograd.Variable(torch.randn(2, batch_size, self.char_hidden_dim // 2))),
                cuda(autograd.Variable(torch.randn(2, batch_size, self.char_hidden_dim // 2))))

    def init_char_embeddings(self, sent_len, batch_size=1):
        return cuda(autograd.Variable(torch.randn(sent_len, batch_size, self.char_hidden_dim)))

    def compute_chars_embeddings(self, chars, lens, wlens, wsorted):

        wc_embeds = self.init_char_embeddings(lens[0], batch_size=chars.size(0))

        for i, (char, slen, wlen, wsort) in enumerate(zip(chars, lens, wlens, wsorted)):

            self.hidden_char = self.init_hidden_char(batch_size=slen)
            char_embeds = self.char_embeddings(char)

            packed_chars = pack_padded_sequence(char_embeds[:slen], sorted(wlen.numpy()[:slen], reverse=True),
                                                batch_first=True)

            packed, c_embeds = self.lstm_char(packed_chars, self.hidden_char)

            word_level_cembeds = torch.cat(c_embeds[0], 1)

            _, wlens_sorted = pad_packed_sequence(packed, batch_first=True)

            reordered = word_level_cembeds.clone()
            for k, s in enumerate(wsort.numpy()[:slen]):
                reordered[s] = word_level_cembeds[k]

            wc_embeds[:slen, i] = reordered

        return wc_embeds

    def get_lstm_feats(self, sentences, lens, chars=None, wlens=None, wsorted=None, test=0, gradient_clipping=0):
        if len(sentences.size()) > 1:
            batch_size = sentences.size(0)
        else:
            batch_size = 1

        self.zero_grad()
        self.hidden = self.init_hidden(batch_size=batch_size)

        w_embeds = self.word_embeddings(sentences)

        # character embeddings
        if self.char:
            assert chars is not None and wlens is not None and wsorted is not None
            wc_embeds = self.compute_chars_embeddings(chars, lens, wlens, wsorted)
            w_embeds = torch.cat((w_embeds,
                                  wc_embeds.view(batch_size, -1, self.char_hidden_dim)),
                                 2)

        # tagger
        if test:
            inputs = (1. - self.dropout) * w_embeds
        else:
            inputs = self.drop(w_embeds)

        packed = pack_padded_sequence(inputs, lens.numpy(), batch_first=True)

        lstm_out, self.hidden = self.lstm(packed, self.hidden)

        if test:
            drops = (1. - self.dropout) * lstm_out.data
        else:
            drops = self.drop(lstm_out.data)

        lstm_feats = self.hidden2tag(drops)

        padded_feats, lens = pad_packed_sequence(PackedSequence(lstm_feats, lstm_out.batch_sizes), batch_first=True)

        return padded_feats

    def viterbi_decode(self, padded_feats, lens):
        if self.crf:
            scores = []
            tag_seqs = []

            for i, l in enumerate(lens):
                score, tag_seq = self.CRF._viterbi_decode(padded_feats[i, :l, :])
                scores.append(score)
                tag_seqs.append(tag_seq)

            return scores, tag_seqs

    def test(self, loader):
        self.volatile = True
        preds = []
        losses = []

        for i, (words, chars, tags, lens_unsorted, wlens, wsorted) in enumerate(tqdm(loader)):

            words, chars, lens, wlens, wsorted, tags, sorted_inds = order(words, chars, lens_unsorted, wlens,
                                                                          wsorted,
                                                                          y=tags)

            words_in = autograd.Variable(cuda(words[:, :lens.numpy()[0]]), volatile=True)
            chars_in = autograd.Variable(cuda(chars[:, :lens.numpy()[0]]), volatile=True)
            targets = autograd.Variable(cuda(tags[:, :lens.numpy()[0]]), volatile=True)

            padded_feats = self.get_lstm_feats(words_in, lens, chars=chars_in, wlens=wlens, wsorted=wsorted, test=1)

            if self.crf:
                neg_log_likelihood = self.CRF.neg_log_likelihood(padded_feats, lens, targets)
                preds_sorted = self.viterbi_decode(padded_feats, lens)[1]

                preds_unsorted = [None] * len(sorted_inds)
                for i, idx in enumerate(sorted_inds):
                    preds_unsorted[idx] = preds_sorted[i]

                preds.append(preds_unsorted)
                losses.append(neg_log_likelihood.cpu().data.numpy())


            else:
                ls = cuda(autograd.Variable(torch.zeros((words.size(0)))))
                for i, feat in enumerate(padded_feats):
                    ls[i] = nn.CrossEntropyLoss()(feat[:lens[i]].contiguous().view(-1, self.tagset_size),
                                                  targets[i][:lens[i]].view(-1))
                losses.append(torch.mean(ls).cpu().data.numpy())
                pred_batch = np.argmax(padded_feats.cpu().data.numpy(), axis=2)

                preds_unsorted = pred_batch.copy()
                for i, idx in enumerate(sorted_inds):
                    preds_unsorted[idx] = pred_batch[i]

                for i, p in enumerate(preds_unsorted):
                    preds.append(preds_unsorted[i, :lens_unsorted[i]].tolist())

        if self.crf:
            cat = []
            for l in preds:
                for p in l:
                    cat.append(p)
            preds = cat

        self.volatile = False

        return preds, np.mean(losses)

    def init_metrics(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.metrics = {"ner": {"precision": [], "recall": [], "f1": [], "accuracy": [], "ent_f1": [], "loss": [],
                                "val_loss_dev": [],
                                "precision_test": [], "recall_test": [], "f1_test": [], "accuracy_test": [],
                                "ent_f1_test": [],
                                "val_loss_test": []}}
        self.best_loss = np.inf
        self.best_f1 = 0

    def reload(self):
        if os.path.exists(self.model_path + "last_state_dict"):
            self.load_state_dict(torch.load(self.model_path + "last_state_dict"))
            if torch.cuda.is_available():
                self = self.cuda()
            with open(self.model_path + "metrics.p", "rb") as file:
                self.metrics = pickle.load(file)
            self.best_loss = np.min(self.metrics["ner"]["val_loss_dev"])
            self.best_f1 = np.max(self.metrics["ner"]["ent_f1"])

        else:
            self.init_metrics()

    def load(self, sents_train, sents_dev, sents_test, tags_train, tags_dev, tags_test, batch_size=1, lower=False,
             sort=True):
        loader_train, sorted_train = loader(sents_train, self.w2idx, self.c2idx, tags_train, batch_size, lower=lower,
                                            sort=sort, shuffle=True)
        loader_dev, sorted_dev = loader(sents_dev, self.w2idx, self.c2idx, tags_dev, batch_size, lower=lower, sort=sort)
        loader_test, sorted_test = loader(sents_test, self.w2idx, self.c2idx, tags_test, batch_size, lower=lower,
                                          sort=sort)

        return loader_train, loader_dev, loader_test, sorted_train, sorted_dev, sorted_test

    def train(self, sents_train, sents_dev, sents_test, tags_train, tags_dev, tags_test, lr, batch_size=1,
              lr_method="SGD", lr_decay=1., epochs=20, eps_noimprov=5, freeze_embeddings=False, lower=False):

        loader_train, loader_dev, loader_test, sorted_train, sorted_dev, sorted_test = \
            self.load(sents_train, sents_dev, sents_test, tags_train, tags_dev, tags_test, batch_size, lower=lower,
                      sort=False)

        tags_dev = [tags_dev[i] for i in sorted_dev]
        tags_test = [tags_test[i] for i in sorted_test]

        sents_dev = [sents_dev[i] for i in sorted_dev]
        sents_test = [sents_test[i] for i in sorted_test]

        if torch.cuda.is_available():
            self.cuda()

        if freeze_embeddings:
            self.word_embeddings.weight.requires_grad = False

        nepoch_noimprov = 0
        for epoch in range(epochs):

            trainable_parameters = filter(lambda p: p.requires_grad, self.parameters())

            if lr_method.lower() == "adam":
                optimizer = optim.Adam(trainable_parameters, lr=lr)
            elif lr_method.lower() == "sgd":
                optimizer = optim.SGD(trainable_parameters, lr=lr)

            print("Epoch %s/%s :" % (epoch + 1, epochs))

            losses = []

            for i, (words, chars, tags, lens, wlens, wsorted) in enumerate(tqdm(loader_train)):

                words, chars, lens, wlens, wsorted, tags, sorted_inds = order(words, chars, lens, wlens, wsorted,
                                                                              y=tags)

                words_in = autograd.Variable(cuda(words[:, :lens.numpy()[0]]))
                chars_in = autograd.Variable(cuda(chars[:, :lens.numpy()[0]]))
                targets = autograd.Variable(cuda(tags[:, :lens.numpy()[0]]))

                padded_feats = self.get_lstm_feats(words_in, lens, chars=chars_in, wlens=wlens, wsorted=wsorted)

                if self.crf:
                    loss = self.CRF.neg_log_likelihood(padded_feats, lens, targets)
                else:
                    ls = cuda(autograd.Variable(torch.zeros((batch_size))))
                    for i, feat in enumerate(padded_feats):
                        # softmax = F.log_softmax(feat[:lens[i]])
                        ls[i] = nn.CrossEntropyLoss()(feat[:lens[i]].contiguous().view(-1, self.tagset_size),
                                                      targets[i][:lens[i]].view(-1))
                    loss = torch.mean(ls)

                loss.backward()
                optimizer.step()

                losses.append(loss.cpu().data.numpy())

            preds_dev, ner_loss_dev = self.test(loader_dev)
            eval_metrics_crf(preds_dev, self.metrics, tags_dev, sents_dev, self.idx2tag, self.model_path,
                             dev=True)

            preds_test, ner_loss_test = self.test(loader_test)
            eval_metrics_crf(preds_test, self.metrics, tags_test, sents_test, self.idx2tag, self.model_path)

            loss_epoch = np.mean(losses)

            print("Loss : NER %s" % (loss_epoch))
            print("Dev loss : NER %s" % (ner_loss_dev))
            print("Test loss : NER %s" % (ner_loss_test))

            self.metrics["ner"]["val_loss_dev"].append(ner_loss_dev)
            self.metrics["ner"]["val_loss_test"].append(ner_loss_test)
            self.metrics["ner"]["loss"].append(loss_epoch)

            torch.save(self.state_dict(), self.model_path + "last_state_dict")

            # Save learning curve
            save_plot(self.metrics, self.model_path)
            with open(self.model_path + "metrics.p", "wb") as file:
                pickle.dump(self.metrics, file)

            if self.metrics["ner"]["ent_f1"][-1] > self.best_f1:
                nepoch_noimprov = 0
                print("New best score on dev.")
                print("Saving model...")
                torch.save(self.state_dict(), self.model_path + "best_state_dict")
                self.best_f1 = self.metrics["ner"]["ent_f1"][-1]

            else:
                nepoch_noimprov += 1
                if nepoch_noimprov > eps_noimprov:
                    break
            lr *= lr_decay

        print("Done")

    def predict(self, sents, batch_size=1):
        loader_test, sorted_ids = loader(sents, self.w2idx, self.c2idx, batch_size=batch_size)

        self.volatile = True

        padded_scores = []
        preds = []

        if not self.crf:
            for i, (words, chars, lens, wlens, wsorted) in tqdm(enumerate(loader_test)):
                words_in = autograd.Variable(cuda(words[:, :lens.numpy()[0]]), volatile=True)
                chars_in = autograd.Variable(cuda(chars[:, :lens.numpy()[0]]), volatile=True)

                scores = self.get_lstm_feats(words_in, lens, chars=chars_in, wlens=wlens, wsorted=wsorted, test=1)

                padded_scores.append(pad_packed_sequence(scores))

            for score, lens in padded_scores:
                pred_batch = np.argmax(score.cpu().data.numpy(), axis=2).transpose()
                for i, p in enumerate(pred_batch):
                    preds.append(pred_batch[i, :lens[i]].tolist())

        else:
            for i, (words, chars, lens, wlens, wsorted) in tqdm(enumerate(loader_test)):
                words_in = autograd.Variable(cuda(words[:, :lens.numpy()[0]]), volatile=True)
                chars_in = autograd.Variable(cuda(chars[:, :lens.numpy()[0]]), volatile=True)

                preds.append(self.forward(words_in, lens, chars=chars_in,
                                          wlens=wlens, wsorted=wsorted, test=1)[1])

            cat = []
            for l in preds:
                for p in l:
                    cat.append(p)
            preds = cat

        self.volatile = False

        return [[self.idx2tag[p] for p in preds[sorted_ids.index(i)]] for i, s in enumerate(sorted_ids)]
