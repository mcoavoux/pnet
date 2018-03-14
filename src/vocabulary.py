#! /usr/bin/python3
import numpy as np


ALPHA = 0.8375  # stochastic replacement constant
UNK = "<UNK>"
UNDEF = "<UNDEF>"
START = "START"
STOP = "STOP"

UNK_I, UNDEF_I, START_I, STOP_I = list(range(4))

class Vocabulary:

    def __init__(self, word_freqs):

        self.word_freqs = word_freqs
        self.words = [UNK, UNDEF, START, STOP] + sorted(word_freqs)
        self.w2i = {w: i for i, w in enumerate(self.words)}

        self.c2i = {}
        self.chars = set()
        for w in self.words:
            self.chars |= set(w)
        self.chars = [UNK, UNDEF, START, STOP] + sorted(self.chars)
        self.c2i = {c: i for i, c in enumerate(self.chars)}

    def save(self, filename):
        out = open("{}_words".format(filename), "w")
        for w in self.words:
            out.write("{}\n".format(w))
        out.close()
        out = open("{}_chars".format(filename), "w")
        for c in self.chars:
            out.write("{}\n".format(c))
        out.close()

    def load(self, filename):
        self.words = []
        ins = open("{}_words".format(filename), "r")
        for line in ins:
            self.words.append(line.strip())
        self.w2i = {w: i for i, w in enumerate(self.words)}
        ins.close()

        self.chars = []
        ins = open("{}_chars".format(filename), "r")
        for line in ins:
            self.chars.append(line.strip())
        self.c2i = {c: i for i, c in enumerate(self.chars)}


    def code_sentence_w(self, sentence, stochastic_replacement=False):
        return [self.code_word(w, stochastic_replacement) for w in sentence]

    def code_sentence_cw(self, sentence, stochastic_replacement=False):
        return [(self.code_word(w, stochastic_replacement),
                 self.code_chars(w)) for w in sentence
                ]

    def code_word(self, w, stochastic_replacement=False):
        if w in {START, STOP, UNDEF, UNK}:
            return self.w2i[w]
        if stochastic_replacement:
            threshold = ALPHA / (ALPHA + self.word_freqs[w])
            if np.random.random() < threshold:
                return UNK_I
            else:
                return self.w2i[w]
        else:
            if w in self.w2i:
                return self.w2i[w]
            else:
                return UNK_I

    def code_chars(self, w):
        if w in {START, STOP, UNDEF, UNK}:
            return [UNDEF_I]
        return [self.c2i[c] if c in self.c2i else UNK_I for c in [START] + list(w) + [STOP]]

    def size_words(self):
        return len(self.words)

    def size_chars(self):
        return len(self.chars)


"""
    TODO: update with new UNK / UNDEF constant
"""

class TypedEncoder:
    def __init__(self, typeid, iterable, add_undef):
        self.typeid = typeid

        add = [UNDEF] if add_undef else []
        add.append(UNK)
        self.i2s = add + sorted(set(iterable))
        self.s2i = {s: i for i, s in enumerate(self.i2s)}

    def save(self, filename):
        out = open(filename, "w")
        out.write("{}\n".format(self.typeid))
        for f in self.i2s:
            out.write("{}\n".format(f))
        out.close()

    def load(self, filename):
        ins = open(filename, "r")
        self.typeid = ins.readline().strip()
        self.i2s = []
        for line in ins:
            self.i2s.append(line.strip())
        self.s2i = {s: i for i, s in enumerate(self.i2s)}

    def __iadd__(self, s):
        if s in self.s2i:
            return
        self.s2i[s] = len(self.i2s)
        self.i2s.append(s)
        return self

    def __iter__(self):
        return iter(self.i2s)

    def code(self, s):
        if s not in self.s2i:
            return TypedEncoder.UNK
        return self.s2i[s]

    def decode(self, i):
        assert(i >= 0 and i <= len(self))
        return self.i2s[i]

    def __contains__(self, s):
        return s in self.s2i

    def __len__(self):
        return len(self.i2s)

    def __str__(self):
        return str(self.i2s)

    def __repr__(self):
        return "<type = {} vocabulary: {}>".format(self.typeid, str(self))
