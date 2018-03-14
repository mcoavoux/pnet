
import nltk.tokenize as tokenizer


class Example:
    def __init__(self, sentence, label, metadata = None):
        self.sentence = sentence
        self.label = label
        
        self.p_sentence = tokenizer.word_tokenize(sentence)
        
        self.metadata = metadata
    
    def get_label(self):
        return self.label
    
    def get_sentence(self):
        return self.p_sentence
    
    def get_aux_labels(self):
        return self.metadata




