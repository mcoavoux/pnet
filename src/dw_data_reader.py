
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



def read_one_json(filename, cmap):
    with open(filename) as json_data:
        d = json.load(json_data)
        json_data.close()
        #text = d['teaser']
        try:
            if "text" not in d:
                return None
            text = d['text']
            if d["categoryName"] in cmap:
                label = cmap[d["categoryName"]]
            else:
                label = len(cmap)
                cmap[d["categoryName"]] = label
            example = Example(text, cmap)
            return example
        except KeyError:
            pprint(d)
            
    
def read_from_folder(folder, categories_map):
    corpus = []
    for filename in os.listdir(folder):
        if filename.endswith(".json"):
            f = "{}/{}".format(folder, filename)
            example = read_one_json(f, categories_map)
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
    train = read_from_folder(train_dir, cmap)
    dev = read_from_folder(dev_dir, cmap)
    test = read_from_folder(test_dir, cmap)
    
    examples = ner.tags_NE(train + dev + test, "dw_corpus", k=k)
    return train, dev, test
    

if __name__ == "__main__":
    
    
    examples  = get_dataset()
    
    for ex in examples:
        print(" ".join(ex.p_sentence))
    examples = ner.tags_NE(examples, "dw_corpus", k=k)



