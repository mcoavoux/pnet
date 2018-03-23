

import xml.etree.ElementTree as ET
import sys
import random
random.seed(10)

from collections import defaultdict

import ner
from example import Example

from xml.sax.saxutils import unescape
from nltk import Tree


def preprocess(line):
    return unescape(line.replace("\\", " "))

def count_NE(dataset):
    counts = defaultdict(int)
    for ex in dataset:
        aux = ex.get_aux_labels()
        if aux is not None:
            for e in aux:
                counts[e] += 1
    return counts

def get_dataset(k = 4):
    
    keys = ["source", "url", "title", "image", "category", "description", "rank", "pubdate"]
    filename = "../datasets/newsspace200.xml"
    
    xml_tree = ET.parse(filename)
    root = xml_tree.getroot()
    
    categories = ["World", "Entertainment", "Sports", "Business"]
    # "Top Stories", "Sci/Tech", "Top News", "Europe", "Health", "Italia", "U.S."]
    label_map = dict(zip(categories, range(len(categories))))
    
    #sources = ["Yahoo Business", "Reuters Business", "Washington Post Business", "BBC News Business"]
    #source_map = dict(zip(sources, range(len(sources))))
    
    
    examples = []
    
    i = 0
    d = []
    for c in root:
        assert(c.tag == keys[i%len(keys)])
        d.append(c.text)
        if len(d) == len(keys):
            if d[4] in label_map:
                description = d[2]
                if d[5] is not None:
                    description +=  "  " + d[5]
                ex = Example(preprocess(description), 
                             label = label_map[d[4]],
                             #metadata = [source_map[d[0]]])
                             )
                examples.append(ex)
            d = []
        i += 1

    examples = ner.tags_NE(examples, "ag_corpus", k=k)
    
    random.shuffle(examples)
    
    l = len(examples) // 10
    test, dev, train = examples[:l], examples[l:2*l], examples[2*l:]

    return train, dev, test



if __name__ == "__main__":
    
    examples  = get_dataset()
    
    for ex in examples:
        print(" ".join(ex.p_sentence))
    
    
