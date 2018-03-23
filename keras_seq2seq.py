#coding: utf-8
import MeCab
import numpy as np
import glob
import os
import pickle
import jaconv
import pathlib
import sys
import requests
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential,model_from_json
from keras.layers.core import Dense, Activation, RepeatVector
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
from keras.callbacks import TensorBoard,ModelCheckpoint,CSVLogger, EarlyStopping,Callback
from keras.utils import plot_model




class History_to_slack(Callback):
    def on_epoch_end(self, epoch, logs={}):
        text="="*10+"\n" \
             +"Epoch : "+str(epoch)+"\n" \
             +"Loss : "+str(logs.get('loss'))+"\n" \
             +"Acc : "+str(logs.get('acc'))+"\n" \
             +"Val_loss : "+str(logs.get('val_loss'))+"\n" \
             +"Val_acc : "+str(logs.get('val_acc'))+"\n" \
             +"="*10
        self.print_slack(text)

    def print_slack(self,text):
        webhook_url = 'https://hooks.slack.com/services/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'
        requests.post(webhook_url,data=json.dumps({'text': text,'username': 'Keras','icon_emoji': u':ghost:','link_names': 1,}))


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
                    if (word not in self.word2id_dic.keys()) and (word != '*'):
                        self.word2id_dic[word] = len(self.word2id_dic)
                        self.id2word_dic[len(self.id2word_dic)] = word
            self.word2id_dic[' '] = len(self.word2id_dic)
            self.id2word_dic[len(self.id2word_dic)] = ' '

        with open(file_name) as sentences: # 各テキストの単語にidを付与し，
            for line in sentences:         # id列をonehotシークエンスに変換
                ids = []
                for i in self.mecab.parse(line).splitlines():
                    word = jaconv.kata2hira(i.split(',')[-1])
                    if word != '*':
                        ids.append(self.word2id_dic[word])
                self.text_onehot_lists.append(ids)

            self.text_max_len = max([len(i) for i in self.text_onehot_lists])
            for id, id_list in enumerate(self.text_onehot_lists):
                pad_list = [self.word2id_dic[' '] for i in range(self.text_max_len-len(id_list))]
                self.text_onehot_lists[id] += pad_list
            self.text_onehot_lists = [self.make_onehot(np.array(i)).tolist() for i in self.text_onehot_lists]

        return self.word2id_dic, self.id2word_dic, self.text_onehot_lists


class DataGenerator:
    '''
    音素ベクトルと対応するテキストのone-hotベクトルを返すgenerator
    '''
    def __init__(self):
        self.reset()

    def one2text(self, onehot_vector):
        self.sentence = []
        for row in onehot_vector:
            self.sentence.append(id2word_dic[int((np.where(row==1)[0]))])
        return ''.join(self.sentence)

    def reset(self):
        self.post_data = []
        self.text_data = []

    def make_data(self, batch_size, directory, text_onehot_lists,stage):
        pr = PostReader()
        while True:
            for path in pathlib.Path(directory).iterdir():
                id, post_vec = pr.read_data(path)
                self.post_data.append(post_vec)
                self.text_data.append(text_onehot_lists[id-1])
                if (stage == 'predict'):
                    print(self.one2text(np.array(text_onehot_lists[id-1])))
                if len(self.post_data) == batch_size:
                    self.post_max_len = max([len(i) for i in self.post_data])
                    for i, post_vec in enumerate(self.post_data):
                        pad_vec = [[0 for j in range(2000)] for k in range(self.post_max_len-len(post_vec))]
                        self.post_data[i] += pad_vec
                    try:
                        inputs = np.array(self.post_data, np.float32)
                        targets = np.array(self.text_data, np.float32)
                    except:
                        print(path)
                        import traceback
                        traceback.print_exc()
                    self.reset()
                    yield inputs, targets


if __name__ == '__main__':

    '''
    学習データ準備
    '''
    data_dir = "./data/" + sys.argv[1]
    ve = VocabExtractor()
    word2id_dic, id2word_dic, text_onehot_lists = ve.extract(data_dir + "/sentences.txt")
    train_datagen = DataGenerator()
    train_dir = pathlib.Path(data_dir + '/train/')
    test_datagen = DataGenerator()
    test_dir = pathlib.Path(data_dir + '/test/')

    epochs = 5
    batch_size = 1

    conf_info ="="*10+"\n" \
               +"学習開始日時："+datetime.now().strftime('%Y年%m月%d日%H時%M分%S秒')+"\n" \
               +"="*10+"\n" \
               +"="*10+"\n" \
               +"教師データ数："+str(len(list(train_dir.iterdir())))+"\n" \
               +"テストデータ数： "+str(len(list(test_dir.iterdir())))+"\n" \
               +"エポック数："+str(epochs)+"\n" \
               +"バッチサイズ："+str(batch_size)+"\n" \
               +"="*10

    print(conf_info)
    history_to_slack = History_to_slack()
    history_to_slack.print_slack(conf_info)


    '''
    テスト時
    '''
    if sys.argv[2] == 'test':
        log_dir = './log/' + sys.argv[3] + '/'
        model_json = open(log_dir + 'model.json').read()
        model = model_from_json(model_json)
        model.load_weights(log_dir + 'weights.hdf5')

        print('Answer')
        print('='*10)

        # 予測結果を出力
        prediction_list = (model.predict_generator(
                   generator=test_datagen.make_data(batch_size,test_dir,text_onehot_lists,'predict'),
                   steps=int(np.ceil(len(list(test_dir.iterdir())) / batch_size))))
        
        print('\nPrediction')
        print('='*10)
        for prediction in prediction_list:
            predict_ids = prediction.argmax(axis=-1)
            pre = ''.join(id2word_dic[i] for i in predict_ids)
            print(pre)
        exit()


    '''
    学習時
    '''
    # ログデータ保存先
    log_dir = './log/' + sys.argv[1] + '_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '/'
    graph_dir = log_dir + 'graph/'
    tensorboard_dir = log_dir + 'tensorboard'
    weight_dir = log_dir + 'weight/'
    os.makedirs(graph_dir,exist_ok=True)
    os.makedirs(tensorboard_dir,exist_ok=True)
    os.makedirs(weight_dir,exist_ok=True)
    
    n_in = 2000 # 入力次元数
    n_hidden = 128 # 隠れ層の次元数
    n_out = 874 # 出力次元数

    model = Sequential()

    # Encoder
    model.add(LSTM(n_hidden, input_shape=(None, n_in)))

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

    # コールバック作成
    tensorboard = TensorBoard(log_dir=tensorboard_dir,histogram_freq=0, write_graph=True)
    checkpoint = ModelCheckpoint(filepath = weight_dir + 'epoch{epoch:03d}-loss{loss:.2f}-acc{acc:.2f}-vloss{val_loss:.2f}-vacc{val_acc:.2f}.hdf5', 
                                                 monitor='val_loss', verbose=1, save_best_only=False, mode='auto')
    csv_logger = CSVLogger(log_dir + 'learn_log.csv', separator=',')
    early_stop = EarlyStopping(patience=20)


    # 学習開始
    history = model.fit_generator(
                   generator=train_datagen.make_data(batch_size,train_dir,text_onehot_lists,'train'), 
                   steps_per_epoch=int(np.ceil(len(list(train_dir.iterdir())) / batch_size)),
                   epochs=epochs,
                   validation_data=test_datagen.make_data(batch_size,test_dir,text_onehot_lists,'test'),
                   validation_steps=int(np.ceil(len(list(test_dir.iterdir())) / batch_size)),
                   callbacks=[tensorboard,checkpoint,csv_logger,early_stop,history_to_slack]
              )




    '''
    学習結果保存処理
    '''

    # 学習済みの重みを保存する
    model.save_weights(log_dir + "weights.hdf5")

    # 学習履歴を保存する
    with open(log_dir + "history.pickle", mode='wb') as f:
        pickle.dump(history.history, f)
    
    end_text="="*10+"\n" \
             +"学習終了日時："+datetime.now().strftime('%Y年%m月%d日%H時%M分%S秒')+"\n" \
             +"="*10
    history_to_slack.print_slack(end_text)
