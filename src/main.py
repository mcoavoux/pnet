from collections import defaultdict

import imdb_data_reader
import trustpilot_data_reader
import ag_data_reader

from classifier import *
from example import *
from bilstm import *
from vocabulary import *

def print_data_distributions(dataset):
    main = defaultdict(int)
    aux = defaultdict(int)
    total = len(dataset)
    for example in dataset:
        label = example.get_label()
        main[label] += 1
        meta = example.get_aux_labels()
        for caracteristic in meta:
            aux[caracteristic] += 1
    
    d_main = np.array(list(main.values()))
    d_aux  = np.array(list(aux.values()))

    assert(d_main.sum() == total)
    dist = d_main / total
    mfb = max(dist)
    print("Distribution, main labels: ", dist, " Most frequent baseline : {}".format(100 * mfb))

    d = d_aux / total
    print("Aux distributions / priors : ", d)


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

class PrModel:
    
    def __init__(self, args, model, trainer, bilstm, main_classifier, aux_classifier, adversary_classifier):
        
        self.args = args
        
        self.model = model
        self.trainer = trainer
        
        self.output_folder = args.output
        self.bilstm = bilstm
        
        self.main_classifier = main_classifier
        
        self.aux_classifiers = aux_classifier
        self.adversary_classifier = adversary_classifier
        
        self.adversary = False

    def get_input(self, example, training):
        prefix = get_demographics_prefix(example) if self.args.use_demographics else []
        encoding, transducting = self.bilstm.build_representations(example.get_sentence(), training=training, prefix = prefix)
        
        if self.adversary:
            hidden_layers = self.main_classifier.compute_output_layer(encoding)[:-1]
            input_adversary = dy.concatenate(hidden_layers)
            input_adversary = dy.nobackprop(input_adversary)
            return input_adversary
        return encoding

    def train_one(self, example, target, classifier):
        input_vec = self.get_input(example, True)
        loss = classifier.get_loss(input_vec, target)
        loss.backward()
        self.trainer.update()

    def predict(self, example, target, classifier):
        input_vec = self.get_input(example, False)
        loss, prediction = classifier.get_loss_and_prediction(input_vec, target)
        return loss, prediction

    def evaluate(self, dataset, targets, classifier, adversary):
        self.adversary = adversary
        loss = 0
        acc = 0
        tot = len(dataset)
        assert(len(targets) == len(dataset))
        self.bilstm.disable_dropout()
        predictions = []
        for i, ex in enumerate(dataset):
            l, p = self.predict(ex, targets[i], classifier)
            predictions.append(p)
            if p == targets[i]:
                acc += 1
            loss += l.value()
        return loss / tot, acc / tot * 100, predictions

    def _train(self, train, dev, epochs, classifier, get_label, adversary):

        lr = args.learning_rate
        dc = args.decay_constant
        
        self.adversary = adversary

        random.shuffle(train)
        sample_train = train[:len(dev)]
        self.trainer.learning_rate = lr
        n_updates = 0

        best = 0
        ibest=0
    
        for epoch in range(epochs):
            random.shuffle(train)
            self.bilstm.set_dropout(0.2)
            for i, example in enumerate(train):
                sys.stderr.write("\r{}%".format(i / len(train) * 100))
                
                self.train_one(example, get_label(example), classifier)
                self.trainer.learning_rate = lr / (1 + n_updates * dc)
                n_updates += 1
            
            sys.stderr.write("\r")
            
            targets_t = [get_label(ex) for ex in sample_train]
            targets_d = [get_label(ex) for ex in dev]
            
            loss_t, acc_t, predictions_t = self.evaluate(sample_train, targets_t, classifier, adversary)
            loss_d, acc_d, predictions_d = self.evaluate(dev, targets_d, classifier, adversary)
            
            Fscore = ""
            if self.adversary:
                ftrain = compute_eval_metrics(classifier.output_size(), targets_t, predictions_t)
                fdev = compute_eval_metrics(classifier.output_size(), targets_d, predictions_d)
                #print(ftrain, fdev)
                Fscore = "F: t = {} d = {}".format(ftrain, fdev)
            
            if acc_d > best:
                best = acc_d
                ibest = epoch
                pref = "ad_" if adversary else ""
                self.model.save("{}/{}model{}".format(self.output_folder, pref, ibest))
            
            print("Epoch {} train: l={:.4f} acc={:.2f} dev: l={:.4f} acc={:.2f} {}".format(epoch, loss_t, acc_t, loss_d, acc_d, Fscore), flush=True)
        
        self.model.populate("{}/{}model{}".format(self.output_folder, pref, ibest))

    def train_main(self, train, dev):
        get_label = lambda ex: ex.get_label()
        self._train(train, dev, args.iterations, self.main_classifier, get_label, False)

    def train_adversary(self, train, dev):
        get_label = lambda ex: ex.get_aux_labels()
        self._train(train, dev, args.iterations_adversary, self.adversary_classifier, get_label, True)

def main(args):
    import dynet as dy
    
    if args.dataset == "ag":
        train, dev, test = ag_data_reader.get_dataset(args.num_NE)
    else:
        train, dev, test = trustpilot_data_reader.get_dataset()
    
    labels_main_task = set([ex.get_label() for ex in train])
    
    labels_adve_task = get_aux_labels(train)
    
    print("Train size: {}".format(len(train)))
    print("Dev size:   {}".format(len(dev)))
    print("Test size:  {}".format(len(test)))
    
    print("Train data distribution")
    print_data_distributions(train)

    print("Dev data distribution")
    print_data_distributions(dev)

    print("Test data distribution")
    print_data_distributions(test)



    model = dy.Model()
    
    if args.use_demographics:
        symbols = ["<g={}>".format(i) for i in [0, 1]] + ["<a={}>".format(i) for i in [0, 1]]
    vocabulary = extract_vocabulary(train)
    bilstm = HierarchicalBiLSTM(args, vocabulary, model)
    input_size = bilstm.size()
    main_classifier = MLP(input_size, len(labels_main_task), args.hidden_layers, args.dim_hidden, dy.rectify, model)
    
    trainer = dy.AdamTrainer(model)
    
    if args.subset:
        train = train[:args.subset]
        dev = dev[:args.subset]

    input_size += args.hidden_layers * args.dim_hidden
    output_size = len(labels_adve_task)
    #if args.adversary_type == "softmax":
        #adversary_classifier = MLP(input_size, output_size, args.hidden_layers, args.dim_hidden, dy.rectify, model)
    #else:
    adversary_classifier = MLP_sigmoid(input_size, output_size, args.hidden_layers, args.dim_hidden, dy.rectify, model)

    #### add adversary classifier
    mod = PrModel(args, model, trainer, bilstm, main_classifier, None, adversary_classifier)
    
    print("Train main task")
    mod.train_main(train, dev)
    targets_test = [ex.get_label() for ex in test]
    loss_test, acc_test, _ = mod.evaluate(test, targets_test, mod.main_classifier, False)
    print("\t Test results : l={} acc={}".format(loss_test, acc_test))
    
    trainer.restart()
    print("Train adversary")
    mod.train_adversary(train, dev)
    targets_test = [ex.get_aux_labels() for ex in test]
    loss_test, acc_test, predictions_test = mod.evaluate(test, targets_test, mod.adversary_classifier, True)
    
    print("\t Adversary Test results : l={} acc={}".format(loss_test, acc_test))
    outsize = mod.adversary_classifier.output_size()
    Fscore = compute_eval_metrics(outsize, targets_test, predictions_test)
    print("\tF          = {} ".format(Fscore))
    
    preds = [set() for _ in targets_test]
    Fscore = compute_eval_metrics(outsize, targets_test, preds)
    print("\tF baseline = {} ".format(Fscore))



    print("Sanity check")
    targets_test = [ex.get_label() for ex in test]
    loss_test, acc_test, _ = mod.evaluate(test, targets_test, mod.main_classifier, False)
    print("\t Test results : l={} acc={}".format(loss_test, acc_test))


if __name__ == "__main__":
    import argparse
    import random
    import numpy as np
    import os
    random.seed(10)
    np.random.seed(10)
    
    usage = """TODO: write usage"""
    
    parser = argparse.ArgumentParser(description = usage, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("output", help="Output folder")
    parser.add_argument("dataset", choices=["ag", "tp"], help="Dataset")
    
    parser.add_argument("--iterations", "-i", type=int, default=20, help="Number of training iterations")
    parser.add_argument("--iterations-adversary", "-I", type=int, default=20, help="Number of training iterations")
    
    parser.add_argument("--decay-constant", type=float, default=1e-7)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--aux", action="store_true", help="Use demographics as aux tasks")
    parser.add_argument("--bidirectional", action="store_true", help="Use a bidirectional lstm instead of unidirectional")
    
    parser.add_argument("--adversary-type", choices=["logistic", "softmax"], default="logistic")

    parser.add_argument("--dynet-seed", type=int, default=4 , help="random seed for dynet (needs to be first argument!)")
    parser.add_argument("--dynet-weight-decay", type=float, default=1e-6, help="Weight decay for dynet")


    parser.add_argument("--dim-char","-c", type=int, default=50, help="Dimension of char embeddings")
    parser.add_argument("--dim-crnn","-C", type=int, default=50, help="Dimension of char lstm")
    parser.add_argument("--dim-word","-w", type=int, default=50, help="Dimension of word embeddings")
    parser.add_argument("--dim-wrnn","-W", type=int, default=50, help="Dimension of word lstm")
    
    parser.add_argument("--use-demographics", "-D", action="store_true", help="use demographic variables as input to bi-lstm")
    
    parser.add_argument("--hidden-layers", "-L", type=int, default=1, help="Number of hidden layers")
    parser.add_argument("--dim-hidden", "-l", type=int, default=50, help="Dimension of hidden layers")
    parser.add_argument("--use-char-lstm", action="store_true", help="Use a character LSTM, [default=false]")
    
    parser.add_argument("--subset", "-S", type=int, default=None, help="Train on a subset of n examples for debugging")
    
    parser.add_argument("--num-NE", "-k", type=int, default=4, help="Number of named entities")

    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    if args.dataset == "ag":
        args.adversary_type = "logistic"
    else:
        args.adversary_type = "softmax"
    
    if "--dynet-seed" not in sys.argv:
        sys.argv.extend(["--dynet-seed", str(args.dynet_seed)])
    #if "--dynet-weight-decay" not in sys.argv:
        #sys.argv.extend(["--dynet-weight-decay", str(args.dynet_weight_decay)])
    main(args)
