import numpy as np
import torch

def cap_feature(s):
    """
    Capitalization feature:
    0 = low caps
    1 = all caps
    2 = first letter caps
    3 = at least one capital (not first letter)
    """
    if s.lower() == s:
        return 0
    elif s.upper() == s:
        return 1
    elif s[0].upper() == s[0]:
        return 2
    else:
        return 3


def prepare_sentence(str_words, w2idx, c2idx, lower=False):
    """
    Prepare a sentence for evaluation.
    """
    def f(x): return x.lower() if lower else x

    words = [w2idx[f(w) if f(w) in w2idx else '<UNK>']
             for w in str_words]
    chars = [[c2idx[c] for c in w if c in c2idx]
             for w in str_words]
    caps = [cap_feature(w) for w in str_words]
    return {
        'str_words': str_words,
        'words': words,
        'chars': chars,
        'caps': caps
    }


def create_dict(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dictionary = {}
    for items in item_list:
        for item in items:
            if item not in dictionary:
                dictionary[item] = 1
            else:
                dictionary[item] += 1
    return dictionary


def create_mapping(dictionary):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dictionary.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}

    return item_to_id, id_to_item

def tag_mapping(tags, tag2idx=None):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    if tag2idx:
        n = len(tag2idx)
        idtags = [[tag2idx[t] for t in s] for s in tags]
        return idtags
    else:
        dictionary = create_dict(tags)
        tag2idx, idx2tag = create_mapping(dictionary)
        n = len(tag2idx)
        print("Found %i unique named entity tags" % len(dictionary))
        idtags = [[tag2idx[t] for t in s] for s in tags]
    return idtags, tag2idx, idx2tag

class CoNLLDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, lens):
        self.data = X
        self.labels = y
        self.lens = lens
        
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.lens[idx]

    def __len__(self):
        return len(self.data)

def pad_list(x, pad_index=0):
    lens = [len(s) for s in x]
    maxlen = max(lens)    
    sorted_indices = sorted(range(len(lens)), key=lambda k: lens[k], reverse=True)
    
    batch = pad_index * torch.ones(len(x), maxlen).long()
    
    for i, idx in enumerate(sorted_indices):
        batch[i, :lens[idx]] = torch.LongTensor(x[idx])
    
    return batch, sorted(lens, reverse=True), sorted_indices
