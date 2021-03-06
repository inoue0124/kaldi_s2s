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
import random
from attention_decoder import AttentionDecoder
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
from keras import backend as K
K.tensorflow_backend.set_session(tf.Session(config=tf.ConfigProto(device_count = {'GPU': 0})))
from keras.activations import softmax
from keras.models import Sequential,model_from_json,Model
from keras.layers import Multiply, multiply, Lambda, Concatenate
from keras.layers.core import Flatten, Dense, Activation, RepeatVector, Permute
from keras.engine.topology import Input
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam,SGD
from keras.callbacks import TensorBoard,ModelCheckpoint,CSVLogger, EarlyStopping,Callback
from keras.utils import plot_model


class MyCheckpoint(Callback):
    def __init__(self, model):
        self.model_to_save = model

    def on_epoch_end(self, epoch, logs=None):
        self.model_to_save.save(weight_dir +'model_at_epoch_%d.h5' % epoch)


class History_to_slack(Callback):
    '''
    学習状況をslackに投稿するクラス
    '''


    def __init__(self,train_step):
        self.train_step = train_step

    def on_epoch_end(self, epoch, logs={}):
        text="="*10+"\n" \
             +"Epoch : "+str(epoch+1)+"/200\n" \
             +"Loss : "+str(logs.get('loss'))+"\n" \
             +"Acc : "+str(logs.get('acc'))+"\n" \
             +"Val_loss : "+str(logs.get('val_loss'))+"\n" \
             +"Val_acc : "+str(logs.get('val_acc'))+"\n" \
             +"="*10
        self.print_slack(text,webhook_url)

        val_dir = pathlib.Path(data_dir + '/val/')

        prediction_list = model.predict_generator(
                   generator=test_datagen.make_data(batch_size,val_dir,text_onehot_lists,'predict'),
                   steps=int(np.ceil(len(list(val_dir.iterdir())) / batch_size)))

        print('\nPrediction')
        print('='*10)
        for prediction in prediction_list:
            predict_ids = prediction.argmax(axis=-1)
            pre = ''.join(id2word_dic[i] for i in predict_ids)
            print(pre)
        

    def on_batch_end(self,batch,logs={}):
        text="Batch : "+str(batch+1)+"/"+str(self.train_step)+"\n"
        #self.print_slack(text,webhook_url)


    def print_slack(self,text,webhook_url):
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
        self.answer=[]

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
            files = [x for x in pathlib.Path(directory).iterdir()]
            random.shuffle(files)
            for path in files:
                id, post_vec = pr.read_data(path)
                self.post_data.append(post_vec)
                #id=random.randint(1,200)
                self.text_data.append(text_onehot_lists[id-1])
                if (stage == 'predict'):
                    self.answer.append(self.one2text(np.array(text_onehot_lists[id-1])))
                    pass
                    #print(self.one2text(np.array(text_onehot_lists[id-1])))
                if len(self.text_data) == batch_size:
                    self.post_max_len = max([len(i) for i in self.post_data])
                    #self.post_max_len = 1500
                    for i, post_vec in enumerate(self.post_data):
                        pad_vec = [[0 for j in range(2000)] for k in range(self.post_max_len-len(post_vec))]
                        self.post_data[i] += pad_vec
                        self.post_data[i] = self.post_data[i]
                    try:
                        inputs = np.array(self.post_data, np.float32)
                        inputs = inputs[:,1::2,:]
                        inputs = inputs[:,1::2,:]
                        targets = np.array(self.text_data, np.float32)
                        #inputs,targets = shuffle(inputs,targets)
                    except:
                        print(path)
                        import traceback
                        traceback.print_exc()
                    self.reset()
                    #print(path)
                    #print(inputs,self.one2text(np.array(targets[0])))
                    #inputs=targets
                    yield inputs, targets


if __name__ == '__main__':

    '''
    学習データ準備
    '''
    data_dir = "./data/" + sys.argv[1]
    ve = VocabExtractor()
    word2id_dic, id2word_dic, text_onehot_lists = ve.extract(data_dir + "/sentences.txt")
    #word2id_dic, id2word_dic, text_onehot_lists = ve.extract(data_dir + "/test.txt")
    train_datagen = DataGenerator()
    train_dir = pathlib.Path(data_dir + '/train/')
    test_datagen = DataGenerator()
    test_dir = pathlib.Path(data_dir + '/test/')
    #from IPython import embed
    #embed()

    epochs = 200
    batch_size = 16

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
    webhook_url = open("./webhook").read()
    history_to_slack = History_to_slack(int(np.ceil(len(list(train_dir.iterdir())) / batch_size)))
    history_to_slack.print_slack(conf_info,webhook_url)

    n_in = 2000 # 入力次元数
    n_hidden = 256 # 隠れ層の次元数
    n_out = 874 # 出力次元数
    #n_out = 2400
    in_seq=ve.text_max_len
    #in_seq=375
    #in_seq=848
    print(ve.text_max_len)


    '''
    テスト時
    '''
    if sys.argv[2] == 'test':

        #Attention2（https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/）
    
        model = Sequential()

        # Encoder
        model.add(BatchNormalization(input_shape=(in_seq,n_out)))
        model.add(LSTM(n_hidden,return_sequences=False))

        # Decoder
        model.add(RepeatVector(ve.text_max_len))
        model.add(AttentionDecoder(n_hidden,n_out))
        model.compile(loss='categorical_crossentropy',
                      #optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                      optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
                      metrics=['accuracy'])
 
        log_dir = sys.argv[3] + '/'
        model_json = open(log_dir + 'model.json').read()
        model = model_from_json(model_json)
        weight_dir = pathlib.Path(log_dir + 'weight/')
        model.load_weights(list(weight_dir.iterdir())[-1])
        val_dir = pathlib.Path(data_dir + '/val/')

        print('Answer')
        print('='*10)

        # 予測結果を出力
        prediction_list = (model.predict_generator(
                   generator=test_datagen.make_data(batch_size,val_dir,text_onehot_lists,'predict'),
                   steps=int(np.ceil(len(list(val_dir.iterdir())) / batch_size))))
        
        print('\nPrediction')
        print('='*10)
        for answer,prediction in zip(test_datagen.answer,prediction_list):
            predict_ids = prediction.argmax(axis=1)
            #print(predict_ids)   
            pre = ''.join(id2word_dic[i] for i in predict_ids)
            print(answer+' --> '+pre)
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
    

    #元々のモデル
    
    
    model = Sequential()

    # Encoder
    model.add(BatchNormalization(input_shape=(None,n_in)))
    model.add(LSTM(n_hidden))

    # Decoder
    model.add(RepeatVector(ve.text_max_len))
    model.add(LSTM(n_hidden, return_sequences=True))

    model.add(TimeDistributed(Dense(n_out)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  #optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
                  metrics=['accuracy'])
    


    #Attention1（https://github.com/GINK03/keras-seq2seq）
    """
    inputs      = Input(shape=(1500, n_in))
    encoded     = LSTM(n_hidden)(inputs)

    inputs_a    = inputs
    inputs_a	= BatchNormalization()(inputs_a)
    a_vector    = Dense(n_hidden, activation='softmax')(Flatten()(inputs_a))
    mul         = multiply([encoded, a_vector]) 
    encoder     = Model(inputs, mul)

    x           = RepeatVector(ve.text_max_len)(mul)
    x           = Bidirectional(LSTM(n_hidden, return_sequences=True))(x)
    decoded     = TimeDistributed(Dense(n_out, activation='softmax'))(x)

    model = Model(inputs, decoded)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy')
    """

    
    #Attention2（https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/）
    """
    model = Sequential()

    # Encoder
    model.add(BatchNormalization(input_shape=(in_seq,n_in)))
    model.add(LSTM(n_hidden,return_sequences=False))

    # Decoder
    model.add(RepeatVector(ve.text_max_len))
    model.add(AttentionDecoder(n_hidden,n_out))
    model.compile(loss='categorical_crossentropy',
                  #optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
                  metrics=['accuracy'])
    
    """

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
    #checkpoint = ModelCheckpoint(filepath = weight_dir + 'epoch{epoch:03d}-loss{loss:.2f}-acc{acc:.2f}-vloss{val_loss:.2f}-vacc{val_acc:.2f}.hdf5', 
    #                                             monitor='val_loss', verbose=1, save_best_only=False, mode='auto')
    checkpoint = MyCheckpoint(model)
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
    history_to_slack.print_slack(end_text,webhook_url)
