import logging
from polyglot.detect import Detector
from ast import literal_eval
from pprint import pprint
from collections import defaultdict
from example import Example

import random
random.seed(10)

"""

Structure of an example

{'country': 'france',
 'gender': 'F',
 'item_type': 'user',
 'location': 'France',
 'profile_text': '',
 'reviews': [{'company_id': 'www.cendriyon.com',
              'date': '2013-10-21T06:39:08.000+00:00',
              'rating': '5',
              'text': ['Franchement rien a dire de negatif! facile pour '
                       'commander,produits repondant a mes attentes,pas du '
                       'tout déçue! Chaussures de bonne qualitée et '
                       'confortables! En un mot SUPER! je vais vite '
                       'recommander encore!'],
              'title': 'simple a commander et efficace !'}],
 'user_id': '5333261'}
"""


# Metadata 
GENDER, BIRTH = 0, 1

map_gender={'F':True, 'M':False}



def bucket_age(birth_date, date):
    birth_date = int(birth_date)
    year = int(date.split("-")[0])
    
    if year - birth_date > 45:
        return False
    if year - birth_date < 35:
        return True
    
    return None


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


def get_raw_data(filename):
    data = []
    with open(filename) as f:
        for line in f:
            obj = literal_eval(line)
            data.append(obj)
    return data


def construct_examples(raw_data, language):
    examples = []
    for o in raw_data:
        d = o['reviews'][0]
        if None in [d['text'], d['rating']]:
            continue
        if d['title'] is None:
            d['title'] = ""
        
        review = d['title'] + " " + " ".join(d['text'])
        
        try:
            lang = Detector(review)
        except:
            logging.info("Unknown language")
            continue
        if lang.language.code != language:
            logging.info("discard {} expected {}".format(lang.language.code, language))
            continue

        if 'gender' in o and 'birth_year' in o:
            if o['gender'] is None or o['birth_year'] is None:
                continue
            gen = map_gender[o['gender']]
            age = bucket_age(o['birth_year'], d['date'])
            
            if age != None:
                
                meta = set()
                if gen:
                    meta.add(GENDER)
                if age:
                    meta.add(BIRTH)
                ex = Example(review.lower(), int(d['rating']) - 1, metadata=meta)
                
                if len(ex.get_sentence()) == 0:
                    continue
                examples.append(ex)
    return examples


def get_dataset(lang):
    lang_map = {"fr": "france",
                "de": "germany",
                "dk": "denmark",
                "us": "united_states",
                "uk": "united_kingdom"}
    
    id_to_lang_code = {"fr": "fr",
                       "de": "de",
                       "dk": "da",
                       "us": "en",
                       "uk": "en"}

    filler = "NUTS-regions"
    if lang == "us":
        filler = "geocoded"
    filename = "../datasets/src/{}.auto-adjusted_gender.{}.jsonl.tmp_filtered".format(lang_map[lang], filler)
    
    raw_data = get_raw_data(filename)
    examples = construct_examples(raw_data, id_to_lang_code[lang])

    examples = get_balanced_distribution(examples)

    #if add_demographics:
        #for ex in examples:
            #s = ex.get_sentence()
            #aux = ex.get_aux_labels()
            #s.append("<G={}>".format(aux[0]))
            #s.append("<A={}>".format(aux[1]))

    random.shuffle(examples)
    seg_size = len(examples) // 10
    test, dev, train = examples[:seg_size], examples[seg_size:seg_size*2], examples[seg_size*2:]
    
    return train, dev, test


if __name__ == "__main__":
    
    #raw_data = get_raw_data("../datasets/src/france.auto-adjusted_gender.NUTS-regions.jsonl.tmp_filtered")
    #examples = construct_examples(raw_data)
    
    for l in ["fr", "de", "dk", "us", "uk"]:
        train, dev, test = get_dataset(l)
        print(l, len(train), len(dev), len(test))
        s = 0
        for ex in train:
            s += len(ex.get_sentence())
        
        print(l, s / len(train))



