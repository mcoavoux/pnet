from collections import defaultdict

import imdb_data_reader
import trustpilot_data_reader

from classifier import *
from example import *
from bilstm import *
from vocabulary import *

def print_data_distributions(dataset):
    main = defaultdict(int)
    aux = [defaultdict(int), defaultdict(int)]
    for example in dataset:
        label = example.get_label()
        main[label] += 1
        meta = example.get_aux_labels()
        for count, v in zip(aux, meta):
            count[v] += 1
    d_main = np.array(list(main.values()))
    d_aux = [np.array(list(v.values())) for v in aux]
    
    dist = d_main / d_main.sum()
    mfb = max(dist)
    print("Distribution, main labels: ", dist, " Most frequent baseline : {}".format(100 * mfb))
    for d in d_aux:
        d = d / d.sum()
        mfb = max(d)
        print("Aux distribution : ", d,  " Most frequent baseline : {}".format(100 * mfb))


def get_demographics_prefix(example):
    aux = example.get_aux_labels()
    return ["<g={}>".format(aux[0]), "<a={}>".format(aux[0])]


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

def train_one(trainer, bilstm, example, classifiers, aux=False, use_demographics = False):
    prefix = get_demographics_prefix(example) if use_demographics else []
    
    encoding, transducting = bilstm.build_representations(example.get_sentence(), training=True, prefix = prefix)
    if not aux:
        target = example.get_label()
        loss = classifiers.get_loss(encoding, target)
    else:
        targets = example.get_aux_labels()
        loss = dy.esum([classifiers[i].get_loss(encoding, t) for i, t in enumerate(targets)])
    
    loss.backward()
    trainer.update()

def evaluate_one(bilstm, example, main_classifier, aux_classifiers, adversary=False, use_demographics = False):
    prefix = get_demographics_prefix(example) if use_demographics else []

    encoding, transducting = bilstm.build_representations(example.get_sentence(), training=False, prefix = prefix)
    
    input = encoding
    if adversary:
        input = dy.concatenate(main_classifier.compute_output_layer(encoding)[:-1])
        input = dy.nobackprop(input)

    
    if main_classifier != None and not adversary:
        loss, prediction = main_classifier.get_loss_and_prediction(input, example.get_label())
        res = [(loss.value(), 1 if example.get_label() == prediction else 0)]
    else:
        loss, prediction = None, None
        res = [(0,0)]
    
    targets = example.get_aux_labels()
    if aux_classifiers is not None:
        for i, c in enumerate(aux_classifiers):
            l, p = c.get_loss_and_prediction(input, targets[i])
            
            acc = 1 if targets[i] == p else 0
            res.append((l.value(), acc))
    return res[0], res[1:]

def evaluate(bilstm, dataset, main_classifier, aux_classifiers, adversary=False, use_demographics = False):
    loss = 0
    acc = 0
    bilstm.disable_dropout()
    if aux_classifiers is not None:
        aux_losses = [0 for _ in aux_classifiers]
        aux_accs = [0 for _ in aux_classifiers]
    
    tot = len(dataset)
    for example in dataset:
        
        main, aux = evaluate_one(bilstm, example, main_classifier, aux_classifiers, adversary, use_demographics = use_demographics)
        if main != (None, None):
            loss += main[0]
            acc += main[1]
        
        if aux_classifiers is not None:
            for i in range(len(aux)):
                aux_losses[i] += aux[i][0]
                aux_accs[i] += aux[i][1]
    if aux_classifiers is not None:
        aux_res = list(zip([l / tot for l in aux_losses], [a / tot * 100 for a in aux_accs]))
    else:
        aux_res = []
    return (loss / tot, acc / tot * 100), aux_res

def train_model(model, output_folder, trainer, bilstm, sentiment_classifier, epochs, train, dev, aux_classifiers=None, lr=0.001, dc=1e-6, use_demographics=False):
    
    random.shuffle(train)
    sample_train = train[:len(dev)]
    trainer.learning_rate = lr
    n_updates = 0

    best = 0
    ibest=0
    
    for epoch in range(epochs):
        random.shuffle(train)
        bilstm.set_dropout(0.2)
        for i, example in enumerate(train):
            sys.stderr.write("\r{}%".format(i / len(train) * 100))
            train_one(trainer, bilstm, example, sentiment_classifier, False, use_demographics = use_demographics)
            
            if aux_classifiers is not None:
                train_one(trainer, bilstm, example, aux_classifiers, True)
            
            trainer.learning_rate = lr / (1 + n_updates * dc)
        
        sys.stderr.write("\r")
        
        main_t, aux_t = evaluate(bilstm, sample_train, sentiment_classifier, aux_classifiers, use_demographics =use_demographics)
        main_d, aux_d = evaluate(bilstm, dev, sentiment_classifier, aux_classifiers, use_demographics =use_demographics)
        
        losst, acct = main_t
        lossd, accd = main_d
        losst, acct, lossd, accd = list(map(lambda x : round(x, 4), [losst, acct, lossd, accd]))
        
        if accd > best:
            best = accd
            ibest=epoch
            model.save("{}/model{}".format(output_folder, ibest))
        
        if aux_classifiers is not None:
            aux_train = " ".join(["l={} a={}".format(round(l, 4), round(a, 4)) for l, a in aux_t])
            aux_dev = " ".join(["l={} a={}".format(round(l, 4), round(a, 4)) for l, a in aux_d])
            
            print("Epoch {} train: l={} acc={} aux={} dev: l={} acc={} aux={}".format(epoch, losst, acct, aux_train, lossd, accd, aux_dev))
        else:
            print("Epoch {} train: l={} acc={} dev: l={} acc={}".format(epoch, losst, acct, lossd, accd))
    
    model.populate("{}/model{}".format(output_folder, ibest))



def train_one_adversary(trainer, bilstm, example, sentiment_classifier, adversary_classifiers, use_demographics):
    prefix = get_demographics_prefix(example) if use_demographics else []
    encoding, transducting = bilstm.build_representations(example.get_sentence(), training=True, prefix=prefix)
    
    input_adversary = dy.concatenate(sentiment_classifier.compute_output_layer(encoding)[:-1])
    input_adversary = dy.nobackprop(input_adversary)
    
    targets = example.get_aux_labels()
    loss = dy.esum([adversary_classifiers[i].get_loss(input_adversary, t) for i, t in enumerate(targets)])
    
    loss.backward()
    trainer.update()


def train_adversary(model, output_folder, trainer, bilstm, sentiment_classifier, adversary_classifiers, epochs, train, dev, lr=0.001, dc=1e-6, use_demographics=False):
    random.shuffle(train)
    sample_train = train[:len(dev)]
    trainer.learning_rate = lr
    n_updates = 0

    #best = [0] * len(adversary_classifiers)
    #ibest= [0] * len(adversary_classifiers)
    best = 0
    ibest = 0

    for epoch in range(epochs):
        random.shuffle(train)
        bilstm.set_dropout(0.2)
        for i, example in enumerate(train):
            sys.stderr.write("\r{}%".format(i / len(train) * 100))
            
            train_one_adversary(trainer, bilstm, example, sentiment_classifier, adversary_classifiers, use_demographics)
            
            trainer.learning_rate = lr / (1 + n_updates * dc)
        
        sys.stderr.write("\r")
        
        main_t, aux_t= evaluate(bilstm, sample_train, sentiment_classifier, adversary_classifiers, adversary=True, use_demographics=use_demographics)
        main_d, aux_d = evaluate(bilstm, dev, sentiment_classifier, adversary_classifiers, adversary=True, use_demographics=use_demographics)
        
        #losst, acct = main_t
        #lossd, accd = main_d
        #losst, acct, lossd, accd = list(map(lambda x : round(x, 4), [losst, acct, lossd, accd]))
        
        sum_score = sum([r[1] for r in aux_d])
        if sum_score > best:
            best = sum_score
            ibest = epoch
            model.save("{}/model_aux{}".format(output_folder, ibest))
        

        aux_train = " ".join(["l={} a={}".format(round(l, 4), round(a, 4)) for l, a in aux_t])
        aux_dev = " ".join(["l={} a={}".format(round(l, 4), round(a, 4)) for l, a in aux_d])
        
        print("Epoch {} train-adv={} dev-adv={}".format(epoch, aux_train, aux_dev))
    
    model.populate("{}/model_aux{}".format(output_folder, ibest))



def main(args):
    import dynet as dy
    
    train, dev, test = trustpilot_data_reader.get_dataset()
    
    
    labels_main_task = set([ex.get_label() for ex in train])
    labels_aux_task = list(zip(*[ex.get_aux_labels() for ex in train]))
    labels_aux_task = [set(l) for l in labels_aux_task]
    
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
    
    sentiment_classifier = MLP(args.dim_wrnn * 2, len(labels_main_task), args.hidden_layers, args.dim_hidden, dy.rectify, model)
    
    
    gender_classifier = MLP(args.dim_wrnn * 2, len(labels_aux_task[0]), args.hidden_layers, args.dim_hidden, dy.rectify, model)
    age_classifier = MLP(args.dim_wrnn * 2, len(labels_aux_task[1]), args.hidden_layers, args.dim_hidden, dy.rectify, model)
    
    trainer = dy.SimpleSGDTrainer(model)
    
    if args.subset:
        train_set = train_set[:args.subset]
        dev_set = dev_set[:args.subset]

    if args.aux:
        l = [gender_classifier, age_classifier]
    else:
        l = None
    
    
    train_model(model, args.output, trainer, bilstm, sentiment_classifier, args.iterations, train, dev, l, args.learning_rate, args.decay_constant, args.use_demographics)
    
    
    main_t, aux_t = evaluate(bilstm, test, sentiment_classifier, l, adversary=False, use_demographics = args.use_demographics)
    losst, acct = main_t
    losst, acct  = list(map(lambda x : round(x, 4), [losst, acct]))
    aux_train = " ".join(["l={} a={}".format(round(l, 4), round(a, 4)) for l, a in aux_t])
    print("\t Test results : l={} acc={} aux={}".format(losst, acct, aux_train))

    input_size = args.dim_wrnn * 2 + args.hidden_layers * args.dim_hidden
    output_sizes = len(labels_aux_task[0]), len(labels_aux_task[1])
    adversary_classifiers = [MLP(input_size, output_sizes[i], args.hidden_layers, args.dim_hidden, dy.rectify, model) for i in [0, 1]]
    
    train_adversary(model, args.output, trainer, bilstm, sentiment_classifier, adversary_classifiers, args.iterations_adversary, train, dev, args.learning_rate, args.decay_constant, args.use_demographics)

    main_t, aux_t = evaluate(bilstm, test, sentiment_classifier, adversary_classifiers, adversary=True, use_demographics = args.use_demographics)
    losst, acct = main_t
    losst, acct  = list(map(lambda x : round(x, 4), [losst, acct]))
    aux_train = " ".join(["l={} a={}".format(round(l, 4), round(a, 4)) for l, a in aux_t])
    print("\t Adversary Test results : l={} acc={} aux={}".format(losst, acct, aux_train))
    
    # Sanity check: see that the main model has same results

    print("Sanity check: remove lines")
    main_t, aux_t = evaluate(bilstm, test, sentiment_classifier, l, adversary=False, use_demographics = args.use_demographics)
    losst, acct = main_t
    losst, acct  = list(map(lambda x : round(x, 4), [losst, acct]))
    aux_train = " ".join(["l={} a={}".format(round(l, 4), round(a, 4)) for l, a in aux_t])
    print("\t Test results : l={} acc={} aux={}".format(losst, acct, aux_train))






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
    parser.add_argument("--iterations", "-i", type=int, default=4, help="Number of training iterations")
    parser.add_argument("--iterations-adversary", "-I", type=int, default=6, help="Number of training iterations")
    parser.add_argument("--decay-constant", type=float, default=1e-6)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--aux", action="store_true", help="Use demographics as aux tasks")

    parser.add_argument("--dynet-seed", type=int, default=3 , help="random seed for dynet (needs to be first argument!)")
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

    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    if "--dynet-seed" not in sys.argv:
        sys.argv.extend(["--dynet-seed", str(args.dynet_seed)])
    #if "--dynet-weight-decay" not in sys.argv:
        #sys.argv.extend(["--dynet-weight-decay", str(args.dynet_weight_decay)])
    main(args)
