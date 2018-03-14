
import sys
import random
random.seed(10)

from example import Example


def get_from(folder):
    
    neg_file = "{}/neg_examples".format(folder)
    pos_file = "{}/pos_examples".format(folder)
    
    examples = []
    
    sys.stderr.write("  Loading negative examples...\n")
    for line in open(neg_file):
        line = line.strip()
        if line:
            examples.append(Example(line, 0))
        if len(examples) > 1000:
            break
    sys.stderr.write("  Done.\n")
    sys.stderr.write("  Loading positive examples...\n")
    for line in open(pos_file):
        line = line.strip()
        if line:
            examples.append(Example(line, 1))
        if len(examples) > 2000:
            break
    sys.stderr.write("  Done.\n")
    return examples

def get_train(root_folder):
    sys.stderr.write("Loading training set...\n")
    res = get_from("{}/train".format(root_folder))
    sys.stderr.write("Done.\n")
    return res

def get_test(root_folder):
    sys.stderr.write("Loading test set...\n")
    res = get_from("{}/test".format(root_folder))
    sys.stderr.write("Done.\n")
    return res

def get_dataset():
    train = imdb_data_reader.get_train("../datasets/aclImdb")
    test = imdb_data_reader.get_test("../datasets/aclImdb")
    
    random.shuffle(train)
    
    l = len(train_corpus)
    k = l // 10 * 9
    train, dev = train[:k], train[k:]

    return train, dev, test

