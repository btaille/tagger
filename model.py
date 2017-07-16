import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from torch_utils import prepare_sequence

def clip_grad(v, min, max):
    v.register_hook(lambda g: g.clamp(min, max))
    return v

class Tagger(nn.Module):
    def __init__(self, w_embeddings, hidden_dim, idx2tag, freeze_embeddings=False,
                 dropout=0.):
        super(Tagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.idx2tag = idx2tag
        self.tagset_size = len(self.idx2tag)

        self.word_embeddings = nn.Embedding(w_embeddings.size(0), w_embeddings.size(1))
        self.word_embeddings.weight = nn.Parameter(w_embeddings)
        self.dropout = dropout
        self.drop = nn.Dropout(self.dropout)

        self.lstm = nn.LSTM(w_embeddings.size(1), hidden_dim // 2, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.hidden = self.init_hidden()

        if freeze_embeddings:
            self.word_embeddings.weight.requires_grad = False

    def init_hidden(self):
        if torch.cuda.is_available():
            return (autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2).cuda()),
                    autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2).cuda()))
        else:
            return (autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)),
                    autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)))

    def init_hidden_char(self):
        if torch.cuda.is_available():
            return (autograd.Variable(torch.randn(1, 1, self.char_hidden_dim // 2).cuda()),
                    autograd.Variable(torch.randn(1, 1, self.char_hidden_dim // 2).cuda()))
        else:
            return (autograd.Variable(torch.randn(1, 1, self.char_hidden_dim // 2)),
                    autograd.Variable(torch.randn(1, 1, self.char_hidden_dim // 2)))

    def init_char_embeddings(self, sent_len):
        if torch.cuda.is_available():
            return autograd.Variable(torch.randn(sent_len, 1, self.char_hidden_dim // 2)).cuda()
        else:
            return autograd.Variable(torch.randn(sent_len, 1, self.char_hidden_dim // 2))

    def forward(self, sentence, test=0):
        w_embeds = self.word_embeddings(sentence)

        # ner
        if test:
            inputs = (1. - self.dropout) * w_embeds
        else:
            inputs = self.drop(w_embeds)

        lstm_out, self.hidden = self.lstm(inputs.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space)

        return tag_scores

    def test(self, sentences, tags):
        self.volatile = True
        preds = []
        losses = []

        for i, (s, t) in enumerate(zip(tqdm(sentences), tags)):
            inputs = prepare_sequence(s, volatile=True)
            targets = prepare_sequence(t, volatile=True)

            scores = self.forward(inputs, test=1)

            preds.append(scores.cpu().data.numpy())
            losses.append(nn.NLLLoss()(scores, targets).cpu().data.numpy())

        self.volatile = False

        return preds, losses


    def test_sentence(self, sentence):
        self.volatile = True
        tag_scores = self.forward(sentence, test=1)
        self.volatile = False

        return [self.idx2tag[i] for m, i in torch.max(tag_scores, 1)]

