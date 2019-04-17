
from collections import defaultdict
from example import Example

import random
random.seed(10)

from trustpilot_data_reader import GENDER, BIRTH, map_gender, get_balanced_distribution

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


def get_dataset():
    examples = read_data("../datasets/blogdataset/dataset")
    
    examples = get_balanced_distribution(examples)

    seg_size = len(examples) // 10
    test, dev, train = examples[:seg_size], examples[seg_size:seg_size*2], examples[seg_size*2:]
    return train, dev, test


if __name__ == "__main__":
    
    train, dev, test = get_dataset()
    
    for t in train:
        print(t.sentence)
    
    





