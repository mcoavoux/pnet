
import _dynet as dy
import random
import numpy as np
import sys

F, B = 0, 1


class HierarchicalBiLSTM:
    
    def __init__(self, args, vocabulary, model):
        
        self.vocabulary = vocabulary
        
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
        
        self.words = model.add_lookup_parameters((vocabulary.size_words(), args.dim_word))
        

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


    def build_representations(self, sentence, training):
        dy.renew_cg()
        coded_sentence = self.vocabulary.code_sentence_cw(sentence, training)
        
        w_init_f = self.wrnn[F].initial_state()
        w_init_b = self.wrnn[B].initial_state()

        lstm_input = self.get_static_representations(coded_sentence)
        
        contextual_embeddings = [
            w_init_f.transduce(lstm_input),
            list(reversed(w_init_b.transduce(reversed(lstm_input))))
        ]

        return (dy.concatenate([contextual_embeddings[F][-1],
                                contextual_embeddings[B][0]]),
                [dy.concatenate(list(fb)) for fb in zip(*contextual_embeddings)])



