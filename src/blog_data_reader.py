
from collections import defaultdict
from example import Example

import random
random.seed(10)

from trustpilot_data_reader import GENDER, BIRTH, map_gender

def read_data(filename):
    
    examples = []
    
    for line in open(filename):
        
        line = line.strip().split("\t")
    
        topic = line[0]
        age = line[1]
        gender = line[2]
        user = line[3]
        text = line[4]
        
        if topic != "None":
            meta = set()
            if age == "1":
                meta.add(0)
            if gender == "f":
                meta.add(1)
            
            topic = int(topic)
            
            examples.append(Example(text, topic, meta))
    
    return examples

def get_balanced_distribution(examples):
    signatures = defaultdict(list)
    for ex in examples:
        meta = ex.get_aux_labels()
        signatures[tuple(meta)].append(ex)
    
    min_num = 10**10
    subcorpora = list(signatures.values())
    for subcorpus in subcorpora:
        if len(subcorpus) < min_num:
            min_num = len(subcorpus)
    
    balanced_dataset = []
    for subcorpus in subcorpora:
        random.shuffle(subcorpus)
        balanced_dataset.extend(subcorpus[:min_num])
    
    random.shuffle(balanced_dataset)
    return balanced_dataset

def get_dataset():
    examples = read_data("../datasets/blogdataset/dataset")
    
    examples = get_balanced_distribution(examples)

    seg_size = len(examples) // 10
    test, dev, train = examples[:seg_size], examples[seg_size:seg_size*2], examples[seg_size*2:]
    return train, dev, test
