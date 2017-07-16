from nltk.corpus.reader.conll import ConllCorpusReader
from time import time


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
