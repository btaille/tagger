from collections import OrderedDict
import os
import argparse

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("-ep", "--epochs", type=int, help="number of epochs", default=30)
parser.add_argument("-lr", "--learning_rate", type=float, help="learning rate", default=1e-2)
parser.add_argument("-opt", "--optimizer", help="optimizer", default="SGD")
parser.add_argument("-hid", "--hidden", type=int, help="dimension of hidden layer", default=100)
parser.add_argument("-w", "--word_dim", type=int, help="dimension of word embeddings", default=100)
parser.add_argument("-d", "--dropout", type=float, help="dropout", default=0.5)

parser.add_argument("-tr", "--train", help="Train set location", default="")
parser.add_argument("-d", "--dev", help="Dev set location", default="")
parser.add_argument("-te", "--test", help="Test set location", default="")

parser.add_argument("-l", "--reload", type=int, help="Reload last saved model", default=0)
parser.add_argument("-p", "--pre_emb", type=int, help="Load pretrained embeddings", default=1)


# Parameters
args = parser.parse_args()
parameters = OrderedDict()

parameters["epochs"] = args.epochs
parameters["lr"] = args.learning_rate
parameters["optimizer"] = args.optimizer
parameters["hidden"] = args.hidden
parameters["word_dim"] = args.word_dim
parameters["dropout"] = args.dropout

parameters["reload"] = args.reload == 1
parameters["pre_emb"] = args.pre_emb == 1

# Check validity of parameters
assert os.path.isfile(args.train)
assert os.path.isfile(args.dev)
assert os.path.isfile(args.test)
assert 0. <= parameters["dropout"] < 1.
assert not parameters['pre_emb'] or os.path.isfile(parameters['pre_emb'])


param_str = "-".join(["%s:%s" % (str(k), str(v)) for (k, v) in parameters.items()]).lower()
print(param_str)
