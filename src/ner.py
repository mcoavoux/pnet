
import ag_data_reader
import os

from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tag import StanfordNERTagger
from collections import defaultdict

#from nltk.tag import corenlp.CoreNLPNERTagger
#from nltk.tag.stanford import CoreNLPNERTagger

#jar = '../tools/stanford-postagger-full-2018-02-27/stanford-postagger.jar'
#model = 'your_path/stanford-postagger-full-2016-10-31/models/english-left3words-distsim.tagger'


#st = StanfordNERTagger('../datasets/recon_backend/named-entity/classifiers/english.all.3class.distsim.crf.ser.gz')
#st = CoreNLPNERTagger(url='http://localhost:9000')
#st = CoreNLPNERTagger()


replace = {"Bush": "George_W._Bush",
           "Kerry":"John_Kerry",
           "Arafat": "Yasser_Arafat"}

def get_NE(example):
    sentence = example.get_sentence()

    ner_sent = NER(sentence)
    
    res = []
    for c in ner_sent:
        if type(c) != tuple:
            label = c.label()
            children = "_".join([a[0] for a in c])
            res.append((label, children))
    
    return res

def NER(sentence):
    if type(sentence) == str:
        sentence = word_tokenize(sentence)
    
    NE = ne_chunk(pos_tag(sentence))
    
    return NE

def save_NER(dataset, final_file):
    out = open(final_file, "w")
    res = []
    for ex in dataset:
        ne = get_NE(ex)
        res.append(ne)
        line = " ".join(["/".join(t) for t in ne])
        out.write("{}\n".format(line))
    return res

def load_NER(dataset, final_file):
    ins = open(final_file, "r")
    res = []
    for line in ins:
        line  = line.strip()
        if not line:
            res.append([])
        else:
            sline = [tuple(e.split("/")) for e in line.split(" ")]
            res.append(sline)
    return res

def tags_NE(dataset, idcorpus, k=10, filter={'PERSON'}, keep_negatives=False):
    final_file = "../tools/ner_{}".format(idcorpus)
    if not os.path.isfile(final_file):
        nes = save_NER(dataset, final_file)
    else:
        nes = load_NER(dataset, final_file)
    
    counts = defaultdict(int)
    for e in nes:
        for i in range(len(e)):
            ne = e[i]
            if ne[0] in filter:
                if ne[1] in replace:
                    ne = ne[0], replace[ne[1]]
                    e[i] = ne
                counts[ne] += 1
    
    k_most_freq = sorted(counts, key = lambda x : counts[x], reverse=True)[:k]
    mapping = {e : i for i, e in enumerate(k_most_freq)}
    
    for e in k_most_freq:
        try:
            print(e, counts[e])
        except:
            print(type(e), type(counts[e]))
    
    newdataset = []
    for example, ne in zip(dataset, nes):
        meta = {mapping[e] for e in ne if e in k_most_freq}
        if len(meta) > 0:
            example.metadata = meta
            newdataset.append(example)
        elif keep_negatives:
            example.metadata = set()
    return newdataset


def NER_stanford(sentence_list, idcorpus):
    final_file = "../tools/tmp_filename_all_{}.ner".format(idcorpus)
    if not os.path.isfile(final_file):
        all_out = open(final_file, "w")
        os.chdir("../tools/stanford-ner-2018-02-27/")
        
        for i, s in enumerate(sentence_list):
            tmpfile = open("../tmp_filename_{}".format(idcorpus), "w")
            tmpfile.write("{}\n".format(s))
            tmpfile.close()
            
            outfile = "../tmp_filename_{}.ner".format(idcorpus)
            os.system("java -mx600m -cp stanford-ner.jar:lib/* edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier classifiers/english.all.3class.distsim.crf.ser.gz -textFile ../tmp_filename_{} > {}".format(idcorpus, outfile))
            
            res = " ".join(open(outfile).read().split("\n"))
            all_out.write("{}\n".format(res))
        
        all_out.close()
        os.chdir("../../src/")
    
    tmpfile = open(final_file)
    res = []
    for line in tmpfile:
        res.append(line.strip())
    return res

if __name__ == "__main__":
    res = NER_stanford([], "ag_corpus")
    extract_most_frequent(res)
