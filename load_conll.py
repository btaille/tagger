from nltk.corpus.reader.conll import ConllCorpusReader
from time import time
import os
import pickle
from tqdm import tqdm

def load_conll03(files=["eng.train", "eng.testa", "eng.testb"], max_len=200):
    start = time()
    columntypes1 = ["words", "pos", "chunk", "ne"]
    columntypes2 = ["words", "pos", "ne", "chunk"]
    conll_reader1 = ConllCorpusReader("data/CoNLL2003/", files, columntypes1)
    conll_reader2 = ConllCorpusReader("data/CoNLL2003/", files, columntypes2)

    words = []
    poses = []
    chunkes = []
    nes = []

    sentences1 = conll_reader1.iob_sents()
    sentences2 = conll_reader2.iob_sents()

    for i, s1 in enumerate(sentences1):
        if not s1 == [] and len(s1) <= max_len:
            w, pos, chunk = zip(*s1)
            _, _, ne = zip(*sentences2[i])
            words.append(list(w))
            poses.append(list(pos))
            chunkes.append(list(chunk))
            nes.append(list(ne))

    print("Loaded CoNLL03 in %s seconds" % (time() - start))

    return words, poses, chunkes, nes


def load_conll00(files=["train.txt, test.txt"], max_len=200):
    start = time()
    columntypes = ["words", "pos", "chunk"]
    conll_reader = ConllCorpusReader("data/CoNLL2000/", files, columntypes)

    words = []
    poses = []
    chunkes = []
    nes = []

    sentences = conll_reader.iob_sents()

    for i, s in enumerate(sentences):
        if not s == [] and len(s) <= max_len:
            w, pos, chunk = zip(*s)
            words.append(list(w))
            poses.append(list(pos))
            chunkes.append(list(chunk))

    print("Loaded CoNLL00 in %s seconds" % (time() - start))

    return words, poses, chunkes


def correct_nes(nes):
    corrected_nes = []
    for tags in nes:
        corrected_tags = []
        previous_prefix = "O"
        previous_suffix = "O"
        for i, t in enumerate(tags):
            if len(t.split("-")) == 2:
                prefix, suffix = t.split("-")
            else:
                prefix, suffix = ("O", "O")

            if previous_prefix == "O" and prefix == "I":
                corrected_tag = "B-" + suffix
            elif previous_prefix in ["I", "B"] and prefix == "I" and not previous_suffix == suffix:
                corrected_tag = "B-" + suffix
            else:
                corrected_tag = t

            corrected_tags.append(corrected_tag)
            previous_prefix = prefix
            previous_suffix = suffix

        corrected_nes.append(corrected_tags)
    return corrected_nes


def clean_conll03(file, output):
    words, poses, chunks, nes = load_conll03(file)

    corrected_chunks = correct_nes(chunks)
    corrected_nes = correct_nes(nes)

    with open("data/CoNLL2003/" + output, "w") as file:
        for i, s in enumerate(words):
            for w, pos, chunk, ne in zip(s, poses[i], corrected_chunks[i], corrected_nes[i]):
                file.write(" ".join([w, pos, chunk, ne]) + "\n")
            file.write("\n")
            
            
def load_internal_conll(files, data_path = "data/wikipedia2/"):
    start = time()  
    
    if os.path.exists(data_path + "_".join(files) + ".p"):
        with open(data_path + "_".join(files) + ".p", "rb") as file:
            data = pickle.load(file)
    else:
        columntypes = ["words", "pos", "chunk"]
        conll_reader = ConllCorpusReader(data_path, files, columntypes)

        data = []
        sentences = conll_reader.iob_sents()
        
        print(len(sentences))

        for s in sentences:
            if not s == []:
                w, ne, link = zip(*s)
                stats = {}
                for label in ["O", "ORG", "PER", "LOC", "VESSEL", "MISC"]:
                    stats[label] = np.sum([int(t == "O") if label == "O" else int(label in t) for t in ne])
                data.append((w, ne, link, stats))

        with open(data_path + "_".join(files) + ".p", "wb") as file:
            pickle.dump(data, file)

    print("Loaded %s in %s seconds" % ('_'.join(files), time() - start))
    return data


def load_internal_wiki(filename, begin=0, end=-1, data_path="data/wikipedia3/"):
    start = time()  
    
    if os.path.exists(data_path + "%s_%s_%s"%(filename, begin, end) + ".p"):
        with open(data_path + "%s_%s_%s"%(filename, begin, end) + ".p", "rb") as file:
            data = pickle.load(file)
                        
    else:        
        with open(data_path + filename, "r", encoding="utf8") as file:
            previous = "\n"
            sentences = []
            tags = []
            current_sent = []
            current_tag = []
                        
            for i, line in enumerate(tqdm(file)):
                if i >= begin:
                    if len(line.split()) > 1 :
                        assert len(line.split()) == 2
                        w, t = line.split()
                        current_sent.append(w)
                        current_tag.append(t)

                    elif len(current_sent):
                        sentences.append(current_sent)
                        tags.append(current_tag)
                        current_sent = []
                        current_tag = []
                        
                if i == end:
                    break
                           
        data = (sentences, tags)
                           
        with open(data_path + "%s_%s_%s"%(filename, begin, end) + ".p", "wb") as file:
            pickle.dump(data, file)

    print("Loaded %s:%s-%s in %s seconds" % (filename, begin, end, time() - start))
    return data


def compute_internal_dicts(filename, data_path="data/wikipedia3/", lower=False):
    start = time() 
    w2idx = {}
    tag2idx = {}
    c2idx = {}
    
    if os.path.exists(data_path + "w2idx_%s_l%s" % (filename, int(lower)) + ".p"):
        with open(data_path + "w2idx_%s_l%s" % (filename, int(lower)) + ".p", "rb") as file:
            w2idx = pickle.load(file)
        with open(data_path + "tag2idx_%s_l%s" % (filename, int(lower)) + ".p", "rb") as file:
            tag2idx = pickle.load(file)
        with open(data_path + "c2idx_%s_l%s" % (filename, int(lower)) + ".p", "rb") as file:
            c2idx = pickle.load(file)
                        
    else:
        with open(data_path + filename, "r", encoding="utf8") as file:
            for line in tqdm(file):
                if len(line.split()) > 1 :
                    assert len(line.split()) == 2
                    w_raw, t = line.split()
                    if lower:
                        w = w_raw.lower()
                    else:
                        w = w_raw
                    if w not in w2idx:
                        w2idx[w] = len(w2idx) + 1
                        for c in w_raw:
                            if c not in c2idx:
                                c2idx[c] = len(c2idx) + 1

                    if t not in tag2idx:
                        tag2idx[t] = len(tag2idx) + 1
        with open(data_path + "w2idx_%s_l%s" % (filename, int(lower)) + ".p", "wb") as file:
            pickle.dump(w2idx, file)
        with open(data_path + "tag2idx_%s_l%s" % (filename, int(lower)) + ".p", "wb") as file:
            pickle.dump(tag2idx, file)
        with open(data_path + "c2idx_%s_l%s" % (filename, int(lower)) + ".p", "wb") as file:
            pickle.dump(c2idx, file)
            
    print("Computed word and tag dict in %s", time() - start)
    return w2idx, c2idx, tag2idx
    