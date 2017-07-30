import re
import operator

def sent2seq(sents, w2idx, idx_unknown=None):
    if idx_unknown is None:
        idx_unknown = max(w2idx.values())
    return [[w2idx[w] if w in w2idx.keys() else idx_unknown for w in sent] for sent in sents]


def sent2chars(sents, c2idx, idx_unknown=None, inv=False):
    if idx_unknown is None:
        idx_unknown = max(c2idx.values())
    if inv:
        return  [[[c2idx[c] if c in c2idx.keys() else idx_unknown for c in w] for w in sent] for sent in sents], [[[c2idx[c] if c in c2idx.keys() else idx_unknown for c in w[::-1]] for w in sent] for sent in sents]
    else:
        return [[[c2idx[c] if c in c2idx.keys() else idx_unknown for c in w] for w in sent] for sent in sents]

def word_index(sents, lower=False, begin_one=True):
    w2idx = {}
    for s in sents:
        for w in s:
            if lower:
                w = w.lower()
            if w not in w2idx:
                if begin_one:
                    w2idx[w] = len(w2idx) + 1
                else:
                    w2idx[w] = len(w2idx)
    idx2w = {v: k for k, v in w2idx.items()}
    return w2idx, idx2w
    
def word_index_lim(sents, lower=False, min_occ=5):
    w2idx = {}
    occs = {}
    for s in sents:
        for w in s:
            if lower:
                w = w.lower()
            if w not in w2idx:
                w2idx[w] = len(w2idx) + 1
                occs[w] = 1
            else:
                occs[w] += 1 

    # sort = sorted(occs.items(), key=operator.itemgetter(1))        

    w2idx = {w:i for w,i in w2idx.items() if occs[w] >= min_occ}
    idx2w = {v: k for k, v in w2idx.items()}
    
    return w2idx, idx2w, occs


def char_index(sents):
    c2idx = {}
    for s in sents:
        for w in s:
            for c in w:
                if c not in c2idx:
                    c2idx[c] = len(c2idx) + 1

    idx2c = {v: k for k, v in c2idx.items()}
    return c2idx, idx2c


def add_unknown_last(w2idx, idx2w):
    if "<UNK>" not in w2idx:
        idx_unknown = max(w2idx.values()) + 1
        idx2w[idx_unknown] = "<UNK>"
        w2idx["<UNK>"] = idx_unknown

    return w2idx, idx2w


def add_unknown(w2idx, idx2w, idx_unknown=0):
    print(idx_unknown)
    if "<UNK>" not in w2idx:
        print("test")
        idx2w[idx_unknown] = "<UNK>"
        print("test")
        w2idx["<UNK>"] = idx_unknown
    
    return w2idx, idx2w
    
def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)