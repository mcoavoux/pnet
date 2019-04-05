
from collections import defaultdict
import sys

from vocabulary import Vocabulary
import imdb_data_reader
import trustpilot_data_reader
import ag_data_reader
import dw_data_reader
import blog_data_reader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import bert

def compute_conditional_baseline(cond_aux, main):
    results = []
    n_examples = sum(main.values())
    p_y = [main[v] / n_examples for v in sorted(main)]
    assert(abs(sum(p_y) - 1.0) < 1e-7)
    for task in cond_aux:
        distributions = [cond_aux[task][label] / main[label] for label in sorted(main)]
        baselines = [max(d, 1 -d) for d in distributions]
        cond_baseline = sum([p * b for p, b in zip(baselines, p_y)])
        results.append(cond_baseline)
    return results

def print_data_distributions(dataset):
    main = defaultdict(int)
    aux = defaultdict(int)
    
    cond_aux = defaultdict(lambda:defaultdict(int))
    
    total = len(dataset)
    for example in dataset:
        label = example.get_label()
        main[label] += 1
        meta = example.get_aux_labels()
        for caracteristic in meta:
            aux[caracteristic] += 1
            cond_aux[caracteristic][label] += 1
    
    d_main = np.array(list(main.values()))
    d_aux  = np.array(list(aux.values()))

    assert(d_main.sum() == total)
    dist = d_main / total
    mfb = max(dist)
    print("Distribution_main_labels: ", dist, " Most frequent baseline : {}".format(100 * mfb))

    d = d_aux / total
    db = [max(c, 1-c) for c in d]
    print("Aux_distributions_priors:   ", "\t".join(map(lambda x : str(round(x,4)), d)))
    print("Aux_distributions_baselines:", "\t".join(map(lambda x : str(round(x,4)), db)))
    
    cond_baselines = compute_conditional_baseline(cond_aux, main)
    #print(cond_baselines)
    print("Aux_distrib_cond_baselines: ", "\t".join(map(lambda x : str(round(x,4)), cond_baselines)))
    return mfb

def get_demographics_prefix(example):
    aux = example.get_aux_labels()
    gen = "F" if trustpilot_data_reader.GENDER in aux else "M"
    age = "O" if trustpilot_data_reader.BIRTH in aux else "U"
    return ["<g={}>".format(gen), "<a={}>".format(age)]


def extract_vocabulary(dataset, add_symbols=None):
    freqs = defaultdict(int)
    for example in dataset:
        s = example.get_sentence()
        for token in s:
            freqs[token] += 1
    if add_symbols is not None:
        for s in add_symbols:
            freqs[s] += 1000
    return Vocabulary(freqs)

def get_aux_labels(examples):
    labels = set()
    for ex in examples:
        for l in ex.get_aux_labels():
            labels.add(l)
    return labels


def compute_eval_metrics(n_tasks, gold, predictions):
    tp = 0
    all_pred = 0
    all_gold = 0
    
    for gs, ps in zip(gold, predictions):
        ctp = len([i for i in gs if i in ps])
        tp += ctp
        
        all_pred += len(ps)
        all_gold += len(gs)
    precision = 0
    recall = 0
    f = 0
    if all_pred != 0:
        precision = tp / all_pred
    if all_gold != 0:
        recall = tp / all_gold
    if precision != 0 and recall != 0:
        f = 2 * precision * recall / (precision + recall)
    
    p = round(precision * 100, 2)
    r = round(recall * 100, 2)
    f = round(f * 100, 2)
    
    
    acc_all = [0] * n_tasks
    for gs, ps in zip(gold, predictions):
        for i in range(n_tasks):
            if (i in gs) == (i in ps):
                acc_all[i] += 1
    
    acc_all = [round(i * 100 / len(gold), 2) for i in acc_all]
    
    return p, r, f, acc_all

class PrivateClassifier(nn.Module):
    def __init__(self, input_size, n_hidden, n_private_labels):
        super(PrivateClassifier, self).__init__()
        self.layers = nn.Sequential(nn.Linear(input_size, n_hidden),
                                    nn.Tanh(),
                                    nn.Linear(n_hidden, n_private_labels),
                                    nn.Sigmoid())

    def forward(self, input_examples, targets=None):
        probs = self.layers(input_examples)
        output = {"probs": probs, "predictions": probs > 0.5}
        if targets is not None:
            output["loss"] = F.binary_cross_entropy(probs, targets)
        return output


def bert_encoder_dataset(corpus, device):
    bert_encoder = bert.BertEncoder()
    bert_encoder.to(device)
    bert_encoder.eval()

    training_examples = []
    with torch.no_grad():
        for i, example in enumerate(corpus):
            v = bert_encoder(example.p_sentence[:500])[0]

            targets = torch.zeros(2, device=device)
            for j in example.get_aux_labels():
                targets[j] = 1

            training_examples.append((v, targets))
            if i % 100 == 0:
                print(f"{i} / {len(corpus)}")

    return training_examples


def main(args):
    get_data = {"ag": lambda : ag_data_reader.get_dataset(args.num_NE),
                "dw": lambda : dw_data_reader.get_dataset(args.num_NE),
                "bl": lambda : blog_data_reader.get_dataset(),
                "tp_fr": lambda : trustpilot_data_reader.get_dataset("fr"),
                "tp_de": lambda : trustpilot_data_reader.get_dataset("de"),
                "tp_dk": lambda : trustpilot_data_reader.get_dataset("dk"),
                "tp_us": lambda : trustpilot_data_reader.get_dataset("us"),
                "tp_uk": lambda : trustpilot_data_reader.get_dataset("uk")}
    
    train, dev, test = get_data[args.dataset]()
    
    labels_main_task = set([ex.get_label() for ex in train])
    labels_main_task.add(0)

    assert(sorted(labels_main_task) == list(range(len(labels_main_task))))
    
    labels_adve_task = get_aux_labels(train)
    
    print("Train size: {}".format(len(train)))
    print("Dev size:   {}".format(len(dev)))
    print("Test size:  {}".format(len(test)))
    
    print("Train data distribution")
    mfb_train = print_data_distributions(train)

    print("Dev data distribution")
    mfb_dev = print_data_distributions(dev)

    print("Test data distribution")
    mfb_test = print_data_distributions(test)

    results = {}

    #if args.use_demographics:
    symbols = ["<g={}>".format(i) for i in ["F", "M"]] + ["<a={}>".format(i) for i in ["U", "O"]]
    vocabulary = extract_vocabulary(train, add_symbols=symbols)


    if args.subset:
        train = train[:args.subset]
        dev = dev[:args.subset]
        test = test[:args.subset]

    device = torch.device("cuda")
    train_bert = bert_encoder_dataset(train, device)

    for i in train_bert[:20]:
        print(i[0].shape)
        print(i[1])

    dev_bert = bert_encoder_dataset(dev, device)
    test_bert = bert_encoder_dataset(test, device)

    input_size = bert.BERT_DIM
    n_private_labels = 2
    n_hidden = args.dim_hidden
    model = PrivateClassifier(input_size, n_hidden, n_private_labels)

    model.to(device)

    optimizer = optim.Adam(model.parameters())

    random.shuffle(train_bert)

    for input, target in train_bert:
        
        output = model(input, target)
        output["loss"].backward()
        print(output["loss"])
        optimizer.step()
    


if __name__ == "__main__":
    import argparse
    import random
    import numpy as np
    import os
    random.seed(10)
    np.random.seed(10)
    
    usage = """Implements the privacy evaluation protocol described in the article.

(i) Trains a classifier to predict text labels (topic, sentiment)
(ii) Generate a dataset with the hidden
  representations of each text {r(x), z} with:
    * z: binary private variables
    * x: text
    * r(x): vector representation of text
(iii) Trains the attacker to predict z from x and evaluates privacy
"""
    
    parser = argparse.ArgumentParser(description = usage, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("output", help="Output folder")
    parser.add_argument("dataset", choices=["ag", "dw", "tp_fr", "tp_de", "tp_dk", "tp_us", "tp_uk", "bl"], help="Dataset. tp=trustpilot, bl=blog")
    
    parser.add_argument("--iterations", "-i", type=int, default=20, help="Number of training iterations")
    #parser.add_argument("--iterations-adversary", "-I", type=int, default=20, help="Number of training iterations for attacker")
    #parser.add_argument("--decay-constant", type=float, default=1e-6)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    #parser.add_argument("--aux", action="store_true", help="Use demographics as aux tasks [not used in article]")
    #parser.add_argument("--bidirectional", action="store_true", help="Use a bidirectional lstm instead of unidirectional")
    #parser.add_argument("--adversary-type", choices=["logistic", "softmax"], default="logistic")
    #parser.add_argument("--dynet-seed", type=int, default=4 , help="random seed for dynet (needs to be first argument!)")
    #parser.add_argument("--dynet-weight-decay", type=float, default=1e-6, help="Weight decay for dynet")

    parser.add_argument("--dim-char","-c", type=int, default=50, help="Dimension of char embeddings")
    parser.add_argument("--dim-crnn","-C", type=int, default=50, help="Dimension of char lstm")
    parser.add_argument("--dim-word","-w", type=int, default=50, help="Dimension of word embeddings")
    parser.add_argument("--dim-wrnn","-W", type=int, default=50, help="Dimension of word lstm")
    
    #parser.add_argument("--use-demographics", "-D", action="store_true", help="use demographic variables as input to bi-lstm [+DEMO setting in article]")
    
    #parser.add_argument("--hidden-layers", "-L", type=int, default=1, help="Number of hidden layers")
    parser.add_argument("--dim-hidden", "-l", type=int, default=50, help="Dimension of hidden layers")
    #parser.add_argument("--use-char-lstm", action="store_true", help="Use a character LSTM, [default=false]")
    
    parser.add_argument("--subset", "-S", type=int, default=None, help="Train on a subset of n examples for debugging")
    parser.add_argument("--num-NE", "-k", type=int, default=4, help="Number of named entities (topic classification only)")

    # Defense methods
    #parser.add_argument("--atraining", action="store_true", help="Adversarial classification defense (multidetasking)")
    #parser.add_argument("--ptraining", action="store_true", help="Declustering defense")
    #parser.add_argument("--alpha", type=float, default=0.01, help="Scaling value declustering")
    #parser.add_argument("--generator", action="store_true", help="Adversarial generation defense")
    #parser.add_argument("--baseline", action="store_true", help="Train a full model on private variables (upper bound for the attacker)")

    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    main(args)


