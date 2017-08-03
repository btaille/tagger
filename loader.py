import numpy as np
import torch
import torch.utils.data
from utils import sent2seq, sent2chars

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

class CoNLLDataset_chars(torch.utils.data.Dataset):
    def __init__(self, X, chars, lens, wlens, wsorted, y=None):
        self.words = X
        self.chars = chars
        self.lens = lens
        self.wlens = wlens
        self.wsorted = wsorted
        self.labels = y
        
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.words[idx], self.chars[idx], self.labels[idx], self.lens[idx], self.wlens[idx], self.wsorted[idx]
        else:
            return self.words[idx], self.chars[idx], self.lens[idx], self.wlens[idx], self.wsorted[idx]
            
    def __len__(self):
        return len(self.words)

def pad_chars(chars, pad_index=0):
    lens_sents = [len(s) for s in chars]
    lens_words = [[len(w) for w in s] for s in chars]
    sorted_sents = sorted(range(len(lens_sents)), key=lambda k: lens_sents[k], reverse=True)
        
    maxlen_sent = max(lens_sents)
    maxlen = max(np.concatenate(lens_words))
    
    
    unrolled = []
    for s in chars:
        for w in s:
            unrolled.append(w)
    
    batch = pad_index * torch.ones(len(chars), int(maxlen_sent), int(maxlen)).long()
    sorted_indices = pad_index * torch.ones(len(chars), int(maxlen_sent)).long()
    wlens = pad_index * torch.ones(len(chars), int(maxlen_sent)).long()
    
    for i, s in enumerate(sorted_sents):
        ordered, _, sorted_ids = pad_list(chars[s], pad_index)
        for j, w in enumerate(ordered):
            batch[i, j, :lens_words[s][sorted_ids[j]]] = torch.LongTensor(w[:lens_words[s][sorted_ids[j]]])
            sorted_indices[i, :lens_sents[s]] = torch.LongTensor(sorted_ids)
            wlens[i, :lens_sents[s]] = torch.LongTensor(lens_words[s])
            
    return batch, wlens, sorted_indices


def loader(sents, w2idx, c2idx, tags=None, batch_size=1, lower=False):
    sequences = sent2seq(sents, w2idx, lower=lower)
    sequences_chars = sent2chars(sents, c2idx)
    
    words, lens, sorted_ids = pad_list(sequences)
    chars, wlens, wsorted_ids = pad_chars(sequences_chars)
    
    if tags is not None:
        labels, _, sorted_ids_tags = pad_list(tags)
        assert sorted_ids == sorted_ids_tags
        
    else:
        labels = None
        
    dataset = CoNLLDataset_chars(words, chars, lens, wlens, wsorted_ids, y=labels)        
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=True)
    
    return loader, sorted_ids