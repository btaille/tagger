import numpy as np
import json
import os
from gensim.models import KeyedVectors

embeddings_path = "word_embeddings/glove.6B/glove.6B.50d_w2vformat.txt"


def load_embeddings(embeddings_path, binary=False):
    assert os.path.isfile(embeddings_path)
    saving_path = embeddings_path + "_embeddings.p"
    vocab_path = embeddings_path + "_vocab.p"

    if not os.path.exists(saving_path):
        print("Loading word embeddings from %s" % embeddings_path)
        w2v = KeyedVectors.load_word2vec_format(embeddings_path, binary=binary)

        weights = w2v.syn0
        np.save(open(saving_path, 'wb'), weights)

        vocab = dict([(k, v.index) for k, v in w2v.vocab.items()])
        with open(vocab_path, 'w') as f:
            f.write(json.dumps(vocab))
        print("Created word embeddings from %s" % embeddings_path)

    else:
        print("Loading from saved embeddings")
        with open(saving_path, 'rb') as f:
            weights = np.load(f)

    vocab = load_vocab(embeddings_path)

    return weights, vocab


def load_vocab(embeddings_path):
    print("Loading vocab")
    vocab_path = embeddings_path + "_vocab.p"

    with open(vocab_path, 'r') as f:
        data = json.loads(f.read())
    word2idx = data
    idx2word = dict([(v, k) for k, v in data.items()])
    return word2idx, idx2word

def convert_glove(embeddings_path):
    assert os.path.isfile(embeddings_path)
    os.system("python3 -m gensim.scripts.glove2word2vec --input  %s --output %s_w2vformat.txt" % (embeddings_path, embeddings_path.split(".txt")[0]))

if __name__ == "__main__":
    w, v = load_embeddings(embeddings_path)