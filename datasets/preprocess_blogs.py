
import glob
from gensim import corpora, models
from nltk.tokenize import RegexpTokenizer
#from stop_words import get_stop_words

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def process_filename(filename):
    stream = open(filename, encoding="utf8")
    result = []
    try:
        for line in stream:
            line = line.strip()
            if line and line[0] != "<":
                result.append(line)
        return result
    except:
        return result
    

class Example:
    def __init__(self, u, a, g, text):
        self.user = u
        self.age = a
        self.gender = g
        self.text = text
        self.topic = None

    def __str__(self):
        return "{}\t{}\t{}\t{}\t{}".format(self.topic, self.age, self.gender, self.user, self.text)

def main():
    age = [1, 3]
    gen = ["m", "f"]
    
    examples = []
    for a in age:
        for g in gen:
            for filename in glob.glob("blogdataset/{}_{}/*".format(g, a)):
                documents = process_filename(filename)
                
                user = filename.split("/")[-1].split(".")[0]
                
                examples += [Example(user, a, g, doc) for doc in documents]
    
    
    preprocessed_docs = []
    
    tokenizer = RegexpTokenizer(r'\w+')
    
    for doc in examples:
        tokenized = tokenizer.tokenize(doc.text.lower())
        preprocessed_docs.append(tokenized)
    
    dictionary = corpora.Dictionary(preprocessed_docs)
    dictionary.filter_extremes(no_below=2, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in preprocessed_docs]

    
    print("Training LDA")
    ldamodel = models.ldamodel.LdaModel(corpus, num_topics=10, id2word = dictionary, passes=20)
    
    for ex, doc in zip(examples, corpus):
        topics = ldamodel.get_document_topics(doc)
        values =[b for _, b in topics]
        if max(values) > 0.8:
            topic = max(topics, key = lambda x: x[1])[0]
            ex.topic = topic
    
    
    out = open("blogdataset/dataset", "w")
    for ex in examples:
        out.write(str(ex) + "\n")
    

if __name__ == "__main__":
    main()