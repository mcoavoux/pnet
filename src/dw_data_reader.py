
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

CLA_FRE = {'Germany': 6837, 'Europe': 5789, 'Business': 3725, 'Culture': 1907, 'Sports': 1246,
           'Politics': 1230, 'World': 1026, 'Bundesliga': 937, 'Syria': 860, 'Terrorism': 815, 
           'Elections': 777, 'Ukraine': 761, 'United States': 719, 'Refugees': 706, 'Conflict': 628, 
           'Asia': 614, 'Environment': 572, 'Afghanistan': 569, 'Music': 538, 'China': 516, 
           'Crime': 510, 'Turkey': 498, 'European Union': 487, 'Auto Industry': 474, 'Aviation': 462, 
           'Human Rights': 458, 'Economy': 445, 'Egypt': 430, 'Africa': 429, 'Russia': 422, 'Health': 417,
           'International Relations': 393, 'Religion': 388, 'Iraq': 381, 'Diplomacy': 381, 'Justice': 359,
           'Middle East': 359, 'Travel': 356, 'Society': 353, 'Court Cases': 352, 'Disasters': 349, 'Greece': 340,
           'Finance': 325, 'Protests': 315, 'Iran': 311, 'Eurozone crisis': 310, 'Energy': 305, 'Banking': 301}

CLASSES = sorted(CLA_FRE, key = lambda x: CLA_FRE[x], reverse=True)[:20]
CLASSES_MAP = {c: i for i, c in enumerate(CLASSES)}



def read_one_json(filename):
    with open(filename) as json_data:
        d = json.load(json_data)
        json_data.close()
        #text = d['teaser']
        try:
            if "text" not in d or len(d["text"].strip().split()) == 0:
                return None
            text = d['text']
            if d["categoryName"] in CLASSES_MAP:
                label = CLASSES_MAP[d["categoryName"]]
            #elif train:
                #label = len(cmap)
                #cmap[d["categoryName"]] = label
            else:
                return None
            example = Example(text, label)
            return example
        except KeyError:
            pprint(d)
            
    
def read_from_folder(folder):
    corpus = []

    for filename in os.listdir(folder):
        if filename.endswith(".json"):
            f = "{}/{}".format(folder, filename)
            example = read_one_json(f)
            if example is None:
                continue
            corpus.append(example)

    # Sort classes by frequency
    #cats = sorted(categories_map, key = lambda x : categories_map[x])
    #labels = [cats[e.get_label()] for e in corpus]
    #counts = defaultdict(int)
    #for l in labels:
        #counts[l] += 1
    #for l in sorted(counts, key = lambda x : counts[x]):
        #print(l, counts[l])
    #exit(0)

    return corpus
    

def get_dataset(k=10):
    folder="../datasets/en_corpora_sets/{}/json/"
    train_dir = folder.format("train")
    dev_dir = folder.format("dev")
    test_dir = folder.format("test")
    
    #cmap = {}
    train = read_from_folder(train_dir)
    dev = read_from_folder(dev_dir)
    test = read_from_folder(test_dir)
    
    ner.tags_NE(train + dev + test, "dw_corpus", k=k, keep_negatives=False)
    
    
    train = [e for e in train if e.get_aux_labels() != None and len(e.get_aux_labels()) > 0]
    dev =   [e for e in dev   if e.get_aux_labels() != None and len(e.get_aux_labels()) > 0]
    test =  [e for e in test  if e.get_aux_labels() != None and len(e.get_aux_labels()) > 0]
    
    labels = ["<UNK>"] + sorted(set([e.get_label() for e in train]))
    
    labels_map = {l : i for i,l in enumerate(labels)}
    
    label_set = set(labels_map.values())
    Max = max(label_set)
    
    assert(Max +1 == len(label_set))

    for e in train + dev + test:
        if e.label in labels_map:
            e.label = labels_map[e.label]
        else:
            e.label = 0
        assert(e.label in label_set)
        assert(e.label <= Max)
    
    return train, dev, test
    

if __name__ == "__main__":
    
    
    examples  = get_dataset()
    
    for ex in examples:
        print(" ".join(ex.p_sentence))
    examples = ner.tags_NE(examples, "dw_corpus", k=k)



