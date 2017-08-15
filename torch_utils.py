import torch
import torch.autograd as autograd


def cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor


def prepare_sequence(seq, volatile=False):
    return autograd.Variable(cuda(torch.LongTensor(seq)), volatile=volatile)


def prepare_sequence_float(seq, volatile=False):
    return autograd.Variable(cuda(torch.FloatTensor(seq)), volatile=volatile)


def tensor(np_array):
    return cuda(torch.Tensor(np_array))
