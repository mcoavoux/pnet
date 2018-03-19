
import ag_data_reader
import os

from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tag import StanfordNERTagger
#from nltk.tag import corenlp.CoreNLPNERTagger
#from nltk.tag.stanford import CoreNLPNERTagger

#jar = '../tools/stanford-postagger-full-2018-02-27/stanford-postagger.jar'
#model = 'your_path/stanford-postagger-full-2016-10-31/models/english-left3words-distsim.tagger'


#st = StanfordNERTagger('../datasets/recon_backend/named-entity/classifiers/english.all.3class.distsim.crf.ser.gz')
#st = CoreNLPNERTagger(url='http://localhost:9000')
#st = CoreNLPNERTagger()

def add_NE(example):
    sentence = example.get_sentence()
    
    ner_sent = NER(sentence)
    
    res = []
    for c in ner_sent:
        if type(c) != tuple:
            label = c.label()
            children = "_".join([a[0] for a in c])
            res.append((label, children))
    
    if len(res) > 0:
        example.metadata = res
        print(res)


def NER(sentence):
    if type(sentence) == str:
        sentence = word_tokenize(sentence)
    
    NE = ne_chunk(pos_tag(sentence))
    #NE = st.tag(sentence)
    
    print(NE)
    return NE

def extract_most_frequent(sentences):
    for sentence in sentences:
        split_sentence = sentence.split()
        toks = [t.split("/") for t in split_sentence]
        
    

def NER_stanford(sentence_list, idcorpus):
    
    if not os.path.isfile("../tools/tmp_filename_{}.ner".format(idcorpus)):
        os.chdir("../tools/stanford-ner-2018-02-27/")
        
        
        last = 0
        for i, s in enumerate(sentence_list):
            tmpfile = open("../tmp_filename_{}_{}".format(idcorpus, 0), "w")
            tmpfile.write("{}\n".format(s))
            
            if i % 1000 == 0 or i == len(sentence_list):
                tmpfile.close()
                os.system("java -mx600m -cp stanford-ner.jar:lib/* edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier classifiers/english.all.3class.distsim.crf.ser.gz -textFile ../tmp_filename_{f}_{i} >> ../tmp_filename_{f}.ner".format(f=idcorpus, i=last))
                tmpfile = open("../tmp_filename_{}_{}".format(idcorpus, i), "w")
                last = i
        tmpfile.close()
        os.chdir("../../src/")
    
    tmpfile = open("../tools/tmp_filename_{}.ner".format(idcorpus))
    res = []
    for line in tmpfile:
        res.append(line.strip())
    tmpfile.close()
    return res

if __name__ == "__main__":
    res = NER_stanford([], "ag_corpus")
    extract_most_frequent(res)
