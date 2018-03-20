#coding: utf-8
import MeCab
import numpy as np
import glob
import os
import pickle
import jaconv
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers.core import Dense, Activation, RepeatVector
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, CSVLogger
from keras.utils import plot_model


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

        return sentence_id, element_list


class VocabExtractor:
    '''
    正解テキスト群から単語辞書を作成し，各テキストのonehot表現を作成するクラス
    '''

    def __init__(self):
        self.mecab = MeCab.Tagger()
        self.word2id_dic = {}
        self.id2word_dic = {}
        self.text_onehot_lists = []

    def make_onehot(self, id_array):
        return np.eye(len(self.word2id_dic))[id_array]

    def extract(self, file_name):
        with open(file_name) as sentences: # word2id，id2wordの辞書を作成
            for line in sentences:
                ids = []
                for i in self.mecab.parse(line).splitlines():
                    word = jaconv.kata2hira(i.split(',')[-1])
                    print (word)
                    if (word not in self.word2id_dic.keys()):
                        self.word2id_dic[word] = len(self.word2id_dic)
                        self.id2word_dic[len(self.id2word_dic)] = word
            self.word2id_dic['PAD'] = len(self.word2id_dic)
            self.id2word_dic[len(self.id2word_dic)] = 'PAD'

        with open(file_name) as sentences: # 各テキストの単語にidを付与し，
            for line in sentences:         # id列をonehotシークエンスに変換
                ids = []
                for i in self.mecab.parse(line).splitlines():
                    word = jaconv.kata2hira(i.split(',')[-1])
                    ids.append(self.word2id_dic[word])
                self.text_onehot_lists.append(ids)

            self.text_max_len = max([len(i) for i in self.text_onehot_lists])
            for id, id_list in enumerate(self.text_onehot_lists):
                pad_list = [self.word2id_dic['PAD'] for i in range(self.text_max_len-len(id_list))]
                self.text_onehot_lists[id] += pad_list
            self.text_onehot_lists = [self.make_onehot(np.array(i)) for i in self.text_onehot_lists]

        return self.word2id_dic, self.id2word_dic, self.text_onehot_lists


class One2Text:
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


class DataArranger:
    '''
    学習用に音素ベクトルと対応するテキストのone-hotベクトルのバッチ（padding済み）を作るクラス
    '''
    def __init__(self):
        pass

    def make_data(self, data_size, file_list, text_onehot_lists):
        pr = PostReader()
        self.post_data = []
        for i, file in enumerate(file_list):
            id, post_vec = pr.read_data(file)
            self.post_data.append(post_vec)
            if i == 0:
                self.text_onehot_data = text_onehot_lists[id-1]
            elif i == 1:
                self.text_onehot_data = np.stack([self.text_onehot_data,text_onehot_lists[id-1]],axis=0)
            elif i == (data_size-1):
                self.text_onehot_data = np.append(self.text_onehot_data, np.reshape(text_onehot_lists[id-1],(1,24,875)),axis=0)
                break
            else:
                self.text_onehot_data = np.append(self.text_onehot_data, np.reshape(text_onehot_lists[id-1],(1,24,875)),axis=0)

        self.post_max_len = max([len(i) for i in self.post_data])
        for i, post_vec in enumerate(self.post_data):
            pad_vec = [[0 for i in range(2000)] for i in range(self.post_max_len-len(post_vec))]
            self.post_data[i] += pad_vec

        return np.array(self.post_data).astype(np.float32), self.text_onehot_data


if __name__ == '__main__':

    '''
    学習データ準備
    '''
    N = 10
    N_train = int(N * 0.9)
    N_validation = N - N_train
    ve = VocabExtractor()
    word2id_dic, id2word_dic, text_onehot_lists = ve.extract("./text/sentences.txt")
    o2t = One2Text(id2word_dic)
    da = DataArranger()
    X, Y = da.make_data(N,sorted(glob.glob("./data/utt*")),text_onehot_lists)
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, train_size=N_train)

    # ログデータ保存先
    log_dir = './log/' + datetime.now().strftime('%Y%m%d_%H%M%S') + '/'
    graph_dir = log_dir + 'graph/'
    tbcb_dir = log_dir + 'tbcb'
    os.makedirs(graph_dir,exist_ok=True)
    os.makedirs(tbcb_dir,exist_ok=True)


    '''
    モデル設定
    '''
    n_in = len(X[0][0]) # 入力次元数
    n_hidden = 128 # 隠れ層の次元数
    n_out = len(Y[0][0]) # 出力次元数

    model = Sequential()

    # Encoder
    model.add(LSTM(n_hidden, input_shape=(da.post_max_len, n_in)))

    # Decoder
    model.add(RepeatVector(ve.text_max_len))
    model.add(LSTM(n_hidden, return_sequences=True))

    model.add(TimeDistributed(Dense(n_out)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
                  metrics=['accuracy'])

    #　モデルを保存する
    model_json = model.to_json()
    with open(log_dir + "model.json", mode='w') as f:
        f.write(model_json)
    plot_model(model, to_file=graph_dir + 'model.png')

    '''
    モデル学習
    '''
    epochs = 30
    batch_size = 2

    # コールバック作成
    tbcb = TensorBoard(log_dir=tbcb_dir,histogram_freq=0, write_graph=True)
    csv_logger = CSVLogger(log_dir + 'learn_log.csv', separator=',')

    # 学習開始
    for epoch in range(epochs):
        history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=1,
                  validation_data=(X_validation, Y_validation),callbacks=[tbcb,csv_logger])

        # 予測結果を出力
        for i in range(3):
            index = np.random.randint(0, N_validation)
            post_vec = X_validation[np.array([index])]
            text = Y_validation[np.array([index])]
            prediction = model.predict_classes(post_vec, verbose=0)

            post_vec = post_vec.argmax(axis=-1)
            text = text.argmax(axis=-1)

            ans = ''.join(id2word_dic[i] for i in text[0])
            pre = ''.join(id2word_dic[i] for i in prediction[0])

            print('-' * 10)
            print('Answer:  ', ans)
            print('Predict:  ', pre)
            print('-' * 10)


    '''
    学習結果保存処理
    '''

    # 学習済みの重みを保存する
    model.save_weights(log_dir + "weights.hdf5")

    # 学習履歴を保存する
    with open(log_dir + "history.pickle", mode='wb') as f:
        pickle.dump(history.history, f)
