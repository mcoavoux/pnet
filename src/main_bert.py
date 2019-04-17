
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


def no_backprop(tensor):
    new = torch.clone(tensor.detach())
    return new

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
    labels = set()
    private_labels = [set(), set()]
    for example in dataset:
        s = example.get_sentence()
        for token in s:
            freqs[token.lower()] += 1
    if add_symbols is not None:
        for s in add_symbols:
            freqs[s] += 1000

    tokens = sorted(freqs, key=lambda x: freqs[x], reverse=True)
    tokens = ["@pad", "@unk"] + tokens
    return tokens, {tok:i for i, tok in enumerate(tokens)}


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
    def __init__(self, input_size, dim_hidden, n_private_labels):
        super(PrivateClassifier, self).__init__()
#        self.layers = nn.Sequential(nn.Linear(input_size, dim_hidden),
#                                    nn.Tanh(),
#                                    nn.Linear(dim_hidden, n_private_labels),
#                                    nn.Sigmoid())
        self.layers = nn.Sequential(nn.Linear(input_size, n_private_labels),
                                    nn.Sigmoid())

    def forward(self, input_examples, targets=None):
        probs = self.layers(input_examples)
        output = {"probs": probs, "predictions": probs > 0.5}
        if targets is not None:
            output["loss"] = F.binary_cross_entropy(probs, targets)
        return output

class PrivateEncoder(nn.Module):
    def __init__(self, voc_size, dim_embeddings, dim_hidden, n_labels, n_hidden, n_private_labels):
        super(PrivateEncoder, self).__init__()
        self.private_classifier = PrivateClassifier(dim_hidden, n_hidden, n_private_labels)

        self.emb = nn.Embedding(voc_size, dim_embeddings)
        self.rnn = nn.LSTM(input_size = dim_embeddings, hidden_size=dim_hidden,
                            num_layers=1, batch_first=True, bidirectional=False)
        
        self.classifier = nn.Sequential(nn.Linear(dim_hidden, n_hidden),
                                        nn.Tanh(),
                                        nn.Linear(n_hidden, n_labels))

        #self.reverse = bert.GradientReversal(1)

    def forward(self, input_examples, target=None, private_targets=None, reverse_fun=None):
        output, (hn, cn) = self.rnn(self.emb(input_examples))
        encoding = output[:, -1, :]

        probs = self.classifier(encoding)

        output = {"probs": probs, "predictions": probs.detach().cpu().numpy().argmax()}

        if target is not None:
            output["loss"] = F.cross_entropy(probs, target)
        
        if reverse_fun is not None:
#            if reverse_fun == no_backprop:
#                output["private"] = {"loss": torch.tensor([0.0], requires_grad=True), "predictions": torch.zeros(2), "probs": torch.ones(2)}
#            else:
             output["private"] = self.private_classifier(reverse_fun(encoding), targets = private_targets)
        else:
            assert not self.training
            output["private"] = self.private_classifier(encoding, targets = private_targets)
        #output["private"] = self.private_classifier(encoding, targets = private_targets)

        return output

    def just_encode(self, input_examples):
        # Just return the output of the LSTM
        output, (hn, cn) = self.rnn(self.emb(input_examples))
        encoding = output[:, -1, :]
        return torch.clone(encoding.detach())

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

def std_encoder_dataset(model, corpus):
    training_examples = []
    with torch.no_grad():
        for example, _, privates in corpus:
            v = model.just_encode(example)
            training_examples.append((v, privates))
    return training_examples

def corpus_to_tensor(corpus, device, vocabulary):
    data = []
    target_set = set()
    for example in corpus:
        low_case = [word.lower() for word in example.p_sentence]
        tokens_id = [vocabulary[word] if word in vocabulary else vocabulary["@unk"] for word in low_case]
        tokens_id = torch.tensor(tokens_id, device=device).view(1, -1)

        private_targets = torch.zeros(2, device=device)
        for j in example.get_aux_labels():
            private_targets[j] = 1
        
        target = torch.tensor([example.label], device=device)

        target_set.add(example.label)

        data.append((tokens_id, target, private_targets))
    return data, target_set


def eval_model(model, dataset):
    model.eval()
    acc = [0,0]
    loss = 0
    with torch.no_grad():
        for x, y in dataset:
            output = model(x, y)
            loss += output["loss"].item()
            predictions = output["predictions"]

            correct = predictions.view(-1).cpu().numpy() == y.cpu().numpy()
            acc[0] += correct[0]
            acc[1] += correct[1]

    d = len(dataset)
    #return {"acc": [acc[0] / d, acc[1] / d], "loss": loss / d}
    return [acc[0] / d, acc[1] / d], loss / d

def eval_model_priv(model, corpus):
    model.eval()
    acc = 0
    loss = 0
    acc_priv = [0, 0]
    loss_priv = 0
    with torch.no_grad():
        for x, y, z in corpus:
            output = model(x, target=y, private_targets=z)
            loss += output["loss"].item()
            prediction = output["predictions"]
            #print(prediction, y)
            if prediction == y:
                acc += 1
            
            if "loss" in output["private"]:
                loss_priv += output["private"]["loss"].item()
                private_predictions = output["private"]["predictions"]
                #print(private_predictions.cpu().view(-1).numpy(), z.cpu().numpy())
                correct = private_predictions.cpu().view(-1).numpy() == z.cpu().numpy()
                #print(correct)
                acc_priv[0] += correct[0]
                acc_priv[1] += correct[1]

    d = len(corpus)
    return acc / d, loss / d, [acc_priv[0] / d, acc_priv[1] / d], loss_priv / d
    #return acc_train, loss_train, acc_priv_train, loss_priv_train


def train_probe(args, device, train_private, dev_private, test_private):
    print("Training probe for private variables")

    private_model = PrivateClassifier(args.W, dim_hidden=args.dim_hidden, n_private_labels=2)
    private_model.to(device)
    optimizer = optim.Adam(private_model.parameters())

    random.shuffle(train_private)
    sample_train_private = train_private[:len(dev_private)]

    best_dev = [0, 0]
    for iteration in range(args.iterations):
        loss = 0
        random.shuffle(train_private)
        private_model.train()
        for input, target in train_private:
            optimizer.zero_grad()
            output = private_model(input, target)
            output["loss"].backward()
            loss += output["loss"].item()
            optimizer.step()

        acc_dev, loss_dev = eval_model(private_model, dev_private)
        acc_train, loss_train = eval_model(private_model, sample_train_private)
        #print([type(i) for i in [iteration, loss, acc_train * 100, loss_train, acc_dev * 100, loss_dev]])
        summary="Epoch {} Loss = {:.3f} train acc {:.2f} {:.2f} loss {:.3f} dev acc {:.2f} {:.2f} loss {:.3f}"
        print(summary.format(iteration, loss, 
                             acc_train[0] * 100, acc_train[1] * 100, loss_train, 
                             acc_dev[0] * 100, acc_dev[1] * 100, loss_dev), flush=True)


        if sum(acc_dev) > sum(best_dev):
            best_dev = [i for i in acc_dev]
            private_model.cpu()
            torch.save(private_model, "{}/private_model".format(args.output))
            print(f"Best so far, epoch {iteration}", flush=True)
            private_model.to(device)

    private_model = torch.load("{}/private_model".format(args.output))
    private_model.to(device)
    private_model.eval()

    acc_test, loss_test = eval_model(private_model, test_private)

    print("Accuracy, test: {:.2f} {:.2f} loss {:.3f}".format(acc_test[0] * 100, acc_test[1] * 100, loss_test), flush=True)


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
    i2tok, tok2i = extract_vocabulary(train, add_symbols=symbols)

    if args.subset:
        train = train[:args.subset]
        dev = dev[:args.subset]
        test = test[:args.subset]

    device = torch.device("cuda")


    if args.mode == "bert":
        train_bert = bert_encoder_dataset(train, device)

        dev_bert = bert_encoder_dataset(dev, device)
        test_bert = bert_encoder_dataset(test, device)

        input_size = bert.BERT_DIM
        n_private_labels = 2
        n_hidden = args.dim_hidden
        model = PrivateClassifier(input_size, n_hidden, n_private_labels)

        model.to(device)

        optimizer = optim.Adam(model.parameters(), weight_decay=0.005)

        random.shuffle(train_bert)

        sample_train_bert = train_bert[:len(dev_bert)]

        best_dev = [0, 0]
        for iteration in range(args.iterations):
            loss = 0
            random.shuffle(train_bert)
            model.train()
            for input, target in train_bert:
                optimizer.zero_grad()
                output = model(input, target)
                output["loss"].backward()
                loss += output["loss"].item()
                optimizer.step()

            acc_dev, loss_dev = eval_model(model, dev_bert)
            acc_train, loss_train = eval_model(model, sample_train_bert)
            #print([type(i) for i in [iteration, loss, acc_train * 100, loss_train, acc_dev * 100, loss_dev]])
            summary="Epoch {} Loss = {:.3f} train acc {:.2f} {:.2f} loss {:.3f} dev acc {:.2f} {:.2f} loss {:.3f}"
            print(summary.format(iteration, loss, 
                                 acc_train[0] * 100, acc_train[1] * 100, loss_train, 
                                 acc_dev[0] * 100, acc_dev[1] * 100, loss_dev), flush=True)


            if sum(acc_dev) > sum(best_dev):
                best_dev = [i for i in acc_dev]
                model.cpu()
                torch.save(model, "{}/model".format(args.output))
                print(f"Best so far, epoch {iteration}", flush=True)
                model.to(device)

        model = torch.load("{}/model".format(args.output))
        model.to(device)
        model.eval()

        acc_test, loss_test = eval_model(model, test_bert)

        print("Accuracy, test: {:.2f} {:.2f} loss {:.3f}".format(acc_test[0] * 100, acc_test[1] * 100, loss_test), flush=True)

    else:

        train_tensor, target_set = corpus_to_tensor(train, device, tok2i)
        dev_tensor, _ = corpus_to_tensor(dev, device, tok2i)
        test_tensor, _ = corpus_to_tensor(test, device, tok2i)


        num_targets = len(target_set)
        assert num_targets == max(target_set)+1
        #def __init__(self, voc_size, dim_embeddings, dim_hidden, n_labels, n_hidden, n_private_labels):
        model = PrivateEncoder(len(tok2i), args.w, args.W, num_targets, n_hidden=1, n_private_labels=2)

        if args.mode == "adv":
            reverse = bert.GradientReversal(args.R)
        elif args.mode == "std":
            reverse = no_backprop
        else:
            reverse = None
        
        model.to(device)

        optimizer = optim.Adam(model.parameters())

        random.shuffle(train_tensor)

        sample_train = train_tensor[:len(dev_tensor)]

        best_dev = 0
        for iteration in range(args.iterations):
            loss = 0
            private_loss = 0
            random.shuffle(train_tensor)
            model.train()
            for input, target, private_target in train_tensor:
                #print(input, target, private_target)
                optimizer.zero_grad()
                output = model(input, target=target, private_targets=private_target, reverse_fun=reverse)
                if args.mode == "adv":
                    full_loss = output["loss"] + output["private"]["loss"]
                else:
                    full_loss = output["loss"]
                full_loss.backward()
                loss += output["loss"].item()
                private_loss += output["private"]["loss"].item()
                optimizer.step()

            acc_dev, loss_dev, acc_priv_dev, loss_priv_dev = eval_model_priv(model, dev_tensor)
            acc_train, loss_train, acc_priv_train, loss_priv_train = eval_model_priv(model, sample_train)

            summary = "Epoch {} Main[l {:.3f} tr a {:.2f} l {:.2f} dev a {:.2f} l {:.2f}] Priv[tr a {:.2f} {:.2f} l {:.3f} dev {:.2f} {:.2f} l {:.3f}]" 
            print(summary.format(iteration,
                                 loss,
                                 acc_train*100, loss_train, acc_dev*100, loss_dev,
                                 acc_priv_train[0]*100, acc_priv_train[1]*100,
                                 loss_priv_train,
                                 acc_priv_dev[0]*100, acc_priv_dev[1]*100,
                                 loss_priv_dev), flush=True)
            
            if acc_dev > best_dev:
                best_dev = acc_dev
                model.cpu()
                torch.save(model, "{}/model".format(args.output))
                print(f"Best so far, epoch {iteration}", flush=True)
                model.to(device)

        print("Training done")
        model = torch.load("{}/model".format(args.output))
        model.to(device)
        model.eval()

        acc_test, loss_test, acc_priv_test, loss_priv_test = eval_model_priv(model, test_tensor)
        
        summary = "Main training test [a {:.2f} l {:.2f}] Priv[a {:.2f} {:.2f} l {:.3f}]"
        print(summary.format(acc_test, loss_test, acc_priv_test[0]*100, acc_priv_test[1]*100, loss_priv_test), flush=True)

        train_private = std_encoder_dataset(model, train_tensor)
        dev_private = std_encoder_dataset(model, dev_tensor)
        test_private = std_encoder_dataset(model, test_tensor)


        train_probe(args, device, train_private, dev_private, test_private)



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
    
    parser = argparse.ArgumentParser(description = usage, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("output", help="Output folder")
    parser.add_argument("dataset", choices=["ag", "dw", "tp_fr", "tp_de", "tp_dk", "tp_us", "tp_uk", "bl"], help="Dataset. tp=trustpilot, bl=blog")
    
    parser.add_argument("--iterations", "-i", type=int, default=30, help="Number of training iterations")
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
    parser.add_argument("-w", type=int, default=50, help="Dimension of word embeddings")
    parser.add_argument("-W", type=int, default=50, help="Dimension of word lstm")
    
    #parser.add_argument("--use-demographics", "-D", action="store_true", help="use demographic variables as input to bi-lstm [+DEMO setting in article]")
    
    #parser.add_argument("--hidden-layers", "-L", type=int, default=1, help="Number of hidden layers")
    parser.add_argument("--dim-hidden", "-l", type=int, default=50, help="Dimension of hidden layers")
    #parser.add_argument("--use-char-lstm", action="store_true", help="Use a character LSTM, [default=false]")
    
    parser.add_argument("--subset", "-S", type=int, default=None, help="Train on a subset of n examples for debugging")
    parser.add_argument("--num-NE", "-k", type=int, default=4, help="Number of named entities (topic classification only)")

    parser.add_argument("--mode", default="bert", choices=["bert", "std", "adv"], help="bert: evaluate leakage with bert pretrained representations. adv: lstm with adversarial training")

    parser.add_argument("-R", type=float, default=1, help="Scale for reversal layer")

    # Defense methods
    #parser.add_argument("--atraining", action="store_true", help="Adversarial classification defense (multidetasking)")
    #parser.add_argument("--ptraining", action="store_true", help="Declustering defense")
    #parser.add_argument("--alpha", type=float, default=0.01, help="Scaling value declustering")
    #parser.add_argument("--generator", action="store_true", help="Adversarial generation defense")
    #parser.add_argument("--baseline", action="store_true", help="Train a full model on private variables (upper bound for the attacker)")

    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    main(args)


