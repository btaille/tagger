import torch
import torch.autograd as autograd


def prepare_sequence(seq, volatile=False):
    if torch.cuda.is_available():
        return autograd.Variable(torch.LongTensor(seq).cuda(), volatile=volatile)
    else:
        return autograd.Variable(torch.LongTensor(seq), volatile=volatile)


def prepare_sequence_float(seq, volatile=False):
    if torch.cuda.is_available():
        return autograd.Variable(torch.FloatTensor(seq).cuda(), volatile=volatile)
    else:
        return autograd.Variable(torch.FloatTensor(seq), volatile=volatile)


def tensor(np_array):
    if torch.cuda.is_available():
        return torch.Tensor(np_array).cuda()
    else:
        return torch.Tensor(np_array)
