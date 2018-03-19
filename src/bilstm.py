
import _dynet as dy
import random
import numpy as np
import sys

F, B = 0, 1


class HierarchicalBiLSTM:
    
    def __init__(self, args, vocabulary, model):
        
        self.vocabulary = vocabulary
        self.bi = args.bidirectional
        
        if args.use_char_lstm:
            self.crnn = [dy.LSTMBuilder(1, args.dim_char, args.dim_crnn, model),
                         dy.LSTMBuilder(1, args.dim_char, args.dim_crnn, model)]
            
            self.chars = model.add_lookup_parameters((vocabulary.size_chars(), args.dim_char))
        else:
            self.crnn = None
            self.chars = None

        dim_input = args.dim_word
        if args.use_char_lstm:
            dim_input += args.dim_crnn * 2

        self.wrnn = [dy.LSTMBuilder(1, dim_input, args.dim_wrnn, model),
                     dy.LSTMBuilder(1, dim_input, args.dim_wrnn, model)]
        
        self.wrnn[0].disable_dropout()
        self.wrnn[0].disable_dropout()
        
        self.words = model.add_lookup_parameters((vocabulary.size_words(), args.dim_word))
        
        
        self._size = args.dim_wrnn *2 if self.bi else args.dim_wrnn

    def get_static_representations(self, coded_sentence):
        
        word_embeddings = [self.words[token[0]] for token in coded_sentence]
        if self.crnn is None:
            return word_embeddings
        else:
            
            char_embeddings = [[self.chars[i] for i in token[1]] for token in coded_sentence]

            c_init_f = self.crnn[F].initial_state()
            c_init_b = self.crnn[B].initial_state()

            return [dy.concatenate([c_init_f.transduce(c_e)[-1],
                                c_init_b.transduce(reversed(c_e))[-1],
                                word_embeddings[i]])
                for i, c_e in enumerate(char_embeddings)
            ]


    def build_representations(self, sentence, training, prefix = []):
        if self.bi:
            return self.build_representations_bi(sentence, training, prefix)
        else:
            return self.build_representations_mono(sentence, training, prefix)

    def build_representations_bi(self, sentence, training, prefix = []):
        dy.renew_cg()
        coded_sentence = self.vocabulary.code_sentence_cw(sentence, training)
        coded_prefix = self.vocabulary.code_sentence_cw(prefix, training)
        
        w_init_f = self.wrnn[F].initial_state()
        w_init_b = self.wrnn[B].initial_state()

        f_lstm_input = self.get_static_representations(coded_prefix + coded_sentence)
        b_lstm_input = self.get_static_representations(coded_prefix + list(reversed(coded_sentence)))
        
        contextual_embeddings = [
            w_init_f.transduce(f_lstm_input),
            list(reversed(w_init_b.transduce(b_lstm_input)))
        ]

        return (dy.concatenate([contextual_embeddings[F][-1],
                                contextual_embeddings[B][0]]),
                [dy.concatenate(list(fb)) for fb in zip(*contextual_embeddings)])

    def build_representations_mono(self, sentence, training, prefix = []):
        dy.renew_cg()
        coded_sentence = self.vocabulary.code_sentence_cw(sentence, training)
        coded_prefix = self.vocabulary.code_sentence_cw(prefix, training)
        
        w_init_f = self.wrnn[F].initial_state()

        f_lstm_input = self.get_static_representations(coded_prefix + coded_sentence)
        
        contextual_embeddings = w_init_f.transduce(f_lstm_input)

        return (contextual_embeddings[-1], contextual_embeddings)

    def size(self):
        return self._size

    def set_dropout(self, v):
        if self.crnn is not None:
            for c in self.crnn:
                c.set_dropout(v)
        for w in self.wrnn:
            w.set_dropout(v)
    
    def disable_dropout(self):
        if self.crnn is not None:
            for c in self.crnn:
                c.disable_dropout()
        for w in self.wrnn:
            w.disable_dropout()



