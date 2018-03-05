# coding: utf-8
import MeCab
import numpy as np

class DataEncorder:
    def __init__(self, file_path):
        self.mecab = MeCab.Tagger()
        self.worddic = {}
        self.indexdic = {}
        self.onehot_vectors = []
        self.encode_data(file_path)
        self.make_indexdic(file_path)

    def encode_data(self, file_path):
        with open(file_path) as sentences:
            for line in sentences:
                for i in self.mecab.parse(line).splitlines():
                    word = i.split('\t')[0]
                    if (i != 'EOS'):
                        if (word not in self.worddic.keys()):
                            self.worddic[word] = len(self.worddic)
        with open(file_path) as sentences:
            for line in sentences:
                ids = []
                for i in self.mecab.parse(line).splitlines():
                    word = i.split('\t')[0]
                    if (i != 'EOS'):
                        ids.append(self.worddic[word])
                self.onehot_vectors.append(self.make_onehot(np.array(ids)))

    def make_onehot(self, id_array):
        return np.eye(len(self.worddic))[id_array]

    def make_indexdic(self, file_path):
        with open(file_path) as sentences:
            for line in sentences:
                for i in self.mecab.parse(line).splitlines():
                    word = i.split('\t')[0]
                    if (i != 'EOS'):
                        if (word not in self.indexdic.values()):
                            self.indexdic[len(self.indexdic)] = word


class DataDecorder:
    def __init__(self, indexdic):
        self.indexdic = indexdic

    def one2text(self, onehot_vector):
        self.sentence = []
        for row in onehot_vector:
            self.sentence.append(self.indexdic[int((np.where(row==1)[0]))])
        return ''.join(self.sentence)

class seq2seq:
    def __init__(self):
        pass


if __name__ == '__main__':
    de = DataEncorder("./data/sentences.txt")
    print(de.indexdic)
