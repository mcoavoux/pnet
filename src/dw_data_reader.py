
import sys
import os
import random
random.seed(10)

from collections import defaultdict

import json
import xml.etree.ElementTree as ET
from xml.sax.saxutils import unescape
from nltk import Tree
from pprint import pprint

import ner
from example import Example



def read_one_json(filename, cmap, train):
    with open(filename) as json_data:
        d = json.load(json_data)
        json_data.close()
        #text = d['teaser']
        try:
            if "text" not in d or len(d["text"].strip().split()) == 0:
                return None
            text = d['text']
            if d["categoryName"] in cmap:
                label = cmap[d["categoryName"]]
            elif train:
                label = len(cmap)
                cmap[d["categoryName"]] = label
            else:
                return None
            example = Example(text, label)
            return example
        except KeyError:
            pprint(d)
            
    
def read_from_folder(folder, categories_map, train):
    corpus = []
    for filename in os.listdir(folder):
        if filename.endswith(".json"):
            f = "{}/{}".format(folder, filename)
            example = read_one_json(f, categories_map, train)
            if example is None:
                continue
            corpus.append(example)
    return corpus
    

def get_dataset(k=10):
    folder="../datasets/en_corpora_sets/{}/json/"
    train_dir = folder.format("train")
    dev_dir = folder.format("dev")
    test_dir = folder.format("test")
    
    cmap = {}
    train = read_from_folder(train_dir, cmap, True)
    dev = read_from_folder(dev_dir, cmap, False)
    test = read_from_folder(test_dir, cmap, False)
    
    ner.tags_NE(train + dev + test, "dw_corpus", k=k, keep_negatives=False)
    
    train = [e for e in train if e.get_aux_labels() != None and len(e.get_aux_labels()) > 0]
    dev = [e for e in dev if e.get_aux_labels() != None and len(e.get_aux_labels()) > 0]
    test = [e for e in test if e.get_aux_labels() != None and len(e.get_aux_labels()) > 0]
    
    labels = set([e.get_label() for e in train + dev + test])
    labels_map = {l : i for i,l in enumerate(labels)}
    for e in train + dev + test:
        e.label = labels_map[e.label]
    
    return train, dev, test
    

if __name__ == "__main__":
    
    
    examples  = get_dataset()
    
    for ex in examples:
        print(" ".join(ex.p_sentence))
    examples = ner.tags_NE(examples, "dw_corpus", k=k)



