import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import codecs
import os


def eval(result, truth, string):
    assert len(result) == len(truth)
    gt = []
    pred = []

    for i, t in enumerate(truth):
        r = result[i]

        z = zip(t, r)

        for tup in z:
            gt.append(int(string in tup[0]))
            pred.append(int(string in tup[1]))

    precision, recall, f1, _ = precision_recall_fscore_support(gt, pred)
    print(string)
    print("Precision : %s" % precision[1])
    print("Recall : %s" % recall[1])
    print("F1 Score : %s" % f1[1])
    print("")


def micro_precision_recall_f1_accuracy(truths, preds):
    assert len(truths) == len(preds)
    tp, fp, fn = ([], [], [])

    for i in range(max(truths)):
        tp.append(np.sum([truths[k] == i and preds[k] == i for k in range(len(truths))]))
        fp.append(np.sum([not truths[k] == i and preds[k] == i for k in range(len(truths))]))
        fn.append(np.sum([truths[k] == i and not preds[k] == i for k in range(len(truths))]))

    pre = float(np.sum(tp)) / (np.sum(tp) + np.sum(fp))
    rec = float(np.sum(tp)) / (np.sum(tp) + np.sum(fn))
    f1 = 2 * pre * rec / (pre + rec)
    acc = float(np.sum(tp)) / len(truths)

    return pre, rec, f1, acc


def eval_metrics(preds, metrics, tags, sents_test_ner, tag2idx, idx2tag, model_path, dev=False):
    task = "ner"
    if len(preds):
        p, r, f, acc = micro_precision_recall_f1_accuracy(
            np.concatenate(tags),
            np.concatenate([np.argmax(pred, axis=1) for pred in preds])
        )

        print("%s : p %s, r %s, f %s, acc %s" % (task, p, r, f, acc))

        if dev:
            metrics[task]["precision"].append(p)
            metrics[task]["recall"].append(r)
            metrics[task]["f1"].append(f)
            metrics[task]["accuracy"].append(acc)

            ner_f1 = eval_ner(sents_test_ner, preds, tags, tag2idx, idx2tag, model_path, name="dev")
            metrics[task]["ent_f1"].append(ner_f1)

        else:
            metrics[task]["precision_test"].append(p)
            metrics[task]["recall_test"].append(r)
            metrics[task]["f1_test"].append(f)
            metrics[task]["accuracy_test"].append(acc)

            ner_f1 = eval_ner(sents_test_ner, preds, tags, tag2idx, idx2tag, model_path)
            metrics[task]["ent_f1_test"].append(ner_f1)


def eval_ner(sents_test_ner, preds, tags, tag2idx, idx2tag, model_path, name="test"):
    ner_preds = [[idx2tag[i] if i in idx2tag.keys() else 'O' for i in np.argmax(p, axis=1)] for p in preds]
    ner_truths = [[idx2tag[i] for i in tags[idx]] for idx in range(len(tags))]

    ner_f1 = conll_eval(sents_test_ner, ner_truths, ner_preds, tag2idx, idx2tag, model_path, name="ner_" + name)

    print("NER f1 : %s" % ner_f1)

    return ner_f1


def eval_metrics_crf(preds, metrics, tags, sents_test_ner, idx2tag, model_path, dev=False):
    tag2idx = {v: k for k, v in idx2tag.items()}
    task = "ner"
    if len(preds):

        p, r, f, acc = micro_precision_recall_f1_accuracy(
            np.concatenate(tags),
            np.concatenate(preds)
        )

        print("%s : p %s, r %s, f %s, acc %s" % (task, p, r, f, acc))

        if dev:
            metrics[task]["precision"].append(p)
            metrics[task]["recall"].append(r)
            metrics[task]["f1"].append(f)
            metrics[task]["accuracy"].append(acc)

            ner_f1 = eval_ner_crf(sents_test_ner, preds, tags, tag2idx, idx2tag, model_path, name="dev")
            metrics[task]["ent_f1"].append(ner_f1)

        else:
            metrics[task]["precision_test"].append(p)
            metrics[task]["recall_test"].append(r)
            metrics[task]["f1_test"].append(f)
            metrics[task]["accuracy_test"].append(acc)

            ner_f1 = eval_ner_crf(sents_test_ner, preds, tags, tag2idx, idx2tag, model_path)
            metrics[task]["ent_f1_test"].append(ner_f1)


def eval_ner_crf(sents_test_ner, preds, tags, tag2idx, idx2tag, model_path, name="test"):
    ner_preds = [[idx2tag[i] if i in idx2tag.keys() else 'O' for i in preds[idx]] for idx in range(len(preds))]
    ner_truths = [[idx2tag[i] for i in tags[idx]] for idx in range(len(tags))]

    ner_f1 = conll_eval(sents_test_ner, ner_truths, ner_preds, tag2idx, idx2tag, model_path, name="ner_" + name)

    print("NER f1 : %s" % ner_f1)

    return ner_f1


def conll_eval(sents, truths, preds, tag_to_id, id_to_tag, model_path, name="test"):
    eval_script = "/media/bruno/DATA/Projects/NER/conlleval"

    n_tags = len(tag_to_id)
    predictions = []
    count = np.zeros((n_tags, n_tags), dtype=np.int32)

    for sent, true_sent, pred_sent in zip(sents, truths, preds):
        assert len(true_sent) == len(pred_sent)
        for i, (y_true, y_pred) in enumerate(zip(true_sent, pred_sent)):
            new_line = " ".join([sent[i], y_true, y_pred])
            predictions.append(new_line)
            count[tag_to_id[y_true], tag_to_id[y_pred]] += 1
        predictions.append("")

    output_path = model_path + "eval.%s.output" % name
    scores_path = model_path + "eval.%s.scores" % name

    with codecs.open(output_path, 'w', 'utf8') as f:
        f.write("\n".join(predictions))

    os.system("perl %s < %s > %s" % (eval_script, output_path, scores_path))

    # CoNLL evaluation results
    eval_lines = [l.rstrip() for l in codecs.open(scores_path, 'r', 'utf8')]
    # for line in eval_lines:
    #     print(line)

    # Remove temp files
    # os.remove(output_path)
    # os.remove(scores_path)
    # Confusion matrix with accuracy for each tag
    print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * n_tags)).format(
        "ID", "NE", "Total",
        *([id_to_tag[i] for i in range(n_tags)] + ["Percent"])
    ))
    for i in range(n_tags):
        print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * n_tags)).format(
            str(i), id_to_tag[i], str(count[i].sum()),
            *([count[i][j] for j in range(n_tags)] +
              ["%.3f" % (count[i][i] * 100. / max(1, count[i].sum()))])
        ))

    # Global accuracy
    print("%i/%i (%.5f%%)" % (
        count.trace(), count.sum(), 100. * count.trace() / max(1, count.sum())
    ))

    # F1 on all entities
    return float(eval_lines[1].strip().split()[-1])


def save_plot(metrics, model_path):
    plt.figure(figsize=(15, 5))

    # NER
    plt.subplot(131)
    plt.plot(metrics["ner"]["loss"])
    plt.plot(metrics["ner"]["val_loss_dev"])
    plt.plot(metrics["ner"]["val_loss_test"])

    plt.title("NER loss")
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(['train', 'dev', 'test'], loc='upper left')

    plt.subplot(132)
    plt.plot(metrics["ner"]["accuracy"])
    plt.plot(metrics["ner"]["accuracy_test"])
    plt.title("NER accuracy")
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.legend(['dev', 'test'], loc='upper left')

    plt.subplot(133)
    plt.plot(metrics["ner"]["ent_f1"])
    plt.plot(metrics["ner"]["ent_f1_test"])
    plt.plot(np.ones(len(metrics["ner"]["ent_f1"])) * 91.2)
    plt.plot(np.ones(len(metrics["ner"]["ent_f1"])) * 90.94)
    plt.title("NER entity F1")
    plt.ylabel('F1')
    plt.xlabel('epochs')
    plt.legend(['dev', 'test', 'best', 'Lample'], loc='lower right')

    plt.savefig(model_path + "graphs.png", dpi=90)
