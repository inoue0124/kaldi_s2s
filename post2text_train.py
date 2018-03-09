# coding: utf-8
import MeCab
import numpy as np
import glob

class PostReader:
    '''
    音素posteriorベクトルをnumpy配列へ読み込むクラス
    '''

    def read_data(self, file_name):
        element_list=[]
        with open(file_name) as sentences:
            for line in sentences:
                element_list.append(line.split())
        sentence_id = int(element_list[0][0].split("_")[2])
        del element_list[-1][-1], element_list[0]
        return sentence_id, np.array(element_list).astype(np.float64)


class DataEncorder:
    '''
    正解テキスト群から単語辞書を作成し，各テキストのonehot表現を作成するクラス
    '''

    def __init__(self):
        self.mecab = MeCab.Tagger()
        self.word2id_dic = {}
        self.id2word_dic = {}
        self.text_onehot_dic = []

    def make_onehot(self, id_array):
        return np.eye(len(self.word2id_dic))[id_array]

    def encode(self, file_name):
        with open(file_name) as sentences: # word2id，id2wordの辞書を作成
            for line in sentences:
                ids = []
                for i in self.mecab.parse(line).splitlines():
                    word = i.split('\t')[0]
                    if (i != 'EOS'):
                        if (word not in self.word2id_dic.keys()):
                            self.word2id_dic[word] = len(self.word2id_dic)
                            self.id2word_dic[len(self.id2word_dic)] = word

        with open(file_name) as sentences: # 各テキストの単語にidを付与し，
            for line in sentences:         # id列をonehotシークエンスに変換
                ids = []
                for i in self.mecab.parse(line).splitlines():
                    word = i.split('\t')[0]
                    if (i != 'EOS'):
                        ids.append(self.word2id_dic[word])
                self.text_onehot_dic.append(self.make_onehot(np.array(ids)))

        return self.word2id_dic, self.id2word_dic, self.text_onehot_dic


class DataDecorder:
    '''
    onehotベクトルに対応するテキストを生成するクラス
    '''

    def __init__(self, id2word_dic):
        self.id2word_dic = id2word_dic

    def one2text(self, onehot_vector):
        self.sentence = []
        for row in onehot_vector:
            self.sentence.append(self.id2word_dic[int((np.where(row==1)[0]))])
        return ''.join(self.sentence)


class seq2seq:
    def __init__(self):
        pass


if __name__ == '__main__':
    de = DataEncorder()
    word2id_dic, id2word_dic, text_onehot_dic = de.encode("./text/sentences.txt")
    dd = DataDecorder(id2word_dic)
    pr = PostReader()
    file_list = sorted(glob.glob("./data/utt*"))
    for file_name in file_list:
        sentence_id, postvec = pr.read_data(file_name)
