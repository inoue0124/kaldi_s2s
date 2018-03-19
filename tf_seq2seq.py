#coding: utf-8
import MeCab
import numpy as np
import glob
import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


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
        #return sentence_id, np.array(element_list).astype(np.float64)
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
                    word = i.split('\t')[0]
                    if (i != 'EOS'):
                        if (word not in self.word2id_dic.keys()):
                            self.word2id_dic[word] = len(self.word2id_dic)
                            self.id2word_dic[len(self.id2word_dic)] = word
            self.word2id_dic['PAD'] = len(self.word2id_dic)
            self.word2id_dic['EOS'] = len(self.word2id_dic)
            self.id2word_dic[len(self.id2word_dic)] = 'PAD'
            self.id2word_dic[len(self.id2word_dic)] = 'EOS'

        with open(file_name) as sentences: # 各テキストの単語にidを付与し，
            for line in sentences:         # id列をonehotシークエンスに変換
                ids = []
                for i in self.mecab.parse(line).splitlines():
                    word = i.split('\t')[0]
                    if (i != 'EOS'):
                        ids.append(self.word2id_dic[word])
                ids.append(self.word2id_dic['EOS'])
                self.text_onehot_lists.append(ids)

            self.max_len = max([len(i) for i in self.text_onehot_lists])
            for id, id_list in enumerate(self.text_onehot_lists):
                pad_list = [self.word2id_dic['PAD'] for i in range(self.max_len-len(id_list))]
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


class Seq2Seq:
    def __init__(self):
        pass

    def make_data(self, data_size, file_list, text_onehot_lists):
        pr = PostReader()
        self.post_data = []
        for i, file in enumerate(file_list):
            id, post_vec = pr.read_data(file)
            self.post_data.append(post_vec)
            if i == 0:
                self.text_onehot_data = text_onehot_lists[id]
            elif i == 1:
                self.text_onehot_data = np.stack([self.text_onehot_data,text_onehot_lists[id]],axis=0)
            elif i == data_size:
                self.text_onehot_data = np.append(self.text_onehot_data, np.reshape(text_onehot_lists[id],(1,24,942)),axis=0)
                break
            else:
                self.text_onehot_data = np.append(self.text_onehot_data, np.reshape(text_onehot_lists[id],(1,24,942)),axis=0)

        self.max_len = max([len(i) for i in self.post_data])
        for i, post_vec in enumerate(self.post_data):
            pad_vec = [[0 for i in range(2000)] for i in range(self.max_len-len(post_vec))]
            self.post_data[i] += pad_vec

        return np.array(self.post_data).astype(np.float64), self.text_onehot_data


    def inference(self, x, y, n_batch, is_training,
              input_digits=None, output_digits=None,
              n_hidden=None, n_out=None):
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.01)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.zeros(shape, dtype=tf.float32)
            return tf.Variable(initial)

        # Encoder
        encoder = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
        state = encoder.zero_state(n_batch, tf.float32)
        encoder_outputs = []
        encoder_states = []

        with tf.variable_scope('Encoder'):
            for t in range(input_digits):
                if t > 0:
                    tf.get_variable_scope().reuse_variables()
                (output, state) = encoder(x[:, t, :], state)
                encoder_outputs.append(output)
                encoder_states.append(state)

        # Decoder
        decoder = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
        state = encoder_states[-1]
        decoder_outputs = [encoder_outputs[-1]]

        # 出力層の重みとバイアスを事前に定義
        V = weight_variable([n_hidden, n_out])
        c = bias_variable([n_out])
        outputs = []

        with tf.variable_scope('Decoder'):
            for t in range(1, output_digits):
                if t > 1:
                    tf.get_variable_scope().reuse_variables()

                if is_training is True:
                    (output, state) = decoder(y[:, t-1, :], state)
                else:
                    # 直前の出力を入力に用いる
                    linear = tf.matmul(decoder_outputs[-1], V) + c
                    out = tf.nn.softmax(linear)
                    outputs.append(out)
                    out = tf.one_hot(tf.argmax(out, -1), depth=output_digits)
                    (output, state) = decoder(out, state)

                decoder_outputs.append(output)

        if is_training is True:
            output = tf.reshape(tf.concat(decoder_outputs, axis=1),
                                [-1, output_digits, n_hidden])

            #linear = tf.einsum('ijk,kl->ijl', output, V) + c
            linear = tf.matmul(output, V) + c
            return tf.nn.softmax(linear)
        else:
            # 最後の出力を求める
            linear = tf.matmul(decoder_outputs[-1], V) + c
            out = tf.nn.softmax(linear)
            outputs.append(out)

            output = tf.reshape(tf.concat(outputs, axis=1),
                                [-1, output_digits, n_out])
            return output


    def loss(self, y, t):
        cross_entropy = \
            tf.reduce_mean(-tf.reduce_sum(
                           t * tf.log(tf.clip_by_value(y, 1e-10, 1.0)),
                           reduction_indices=[1]))
        return cross_entropy


    def training(self, loss):
        optimizer = \
            tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)
        train_step = optimizer.minimize(loss)
        return train_step


    def accuracy(self, y, t):
        correct_prediction = tf.equal(tf.argmax(y, -1), tf.argmax(t, -1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy



if __name__ == '__main__':

    N = 100
    N_train = int(N * 0.9)
    N_validation = N - N_train
    ve = VocabExtractor()
    word2id_dic, id2word_dic, text_onehot_lists = ve.extract("./text/sentences.txt")
    o2t = One2Text(id2word_dic)
    seq2seq = Seq2Seq()
    X, Y = seq2seq.make_data(N,sorted(glob.glob("./data/utt*")),text_onehot_lists)
    X_train, X_validation, Y_train, Y_validation = \
        train_test_split(X, Y, train_size=N_train)


    # モデル設定
    n_in = len(X[0][0])
    n_hidden = 128
    n_out = len(Y[0][0])

    x = tf.placeholder(tf.float32, shape=[None, seq2seq.max_len, n_in])
    t = tf.placeholder(tf.float32, shape=[None, ve.max_len, n_out])
    n_batch = tf.placeholder(tf.int32, shape=[])
    is_training = tf.placeholder(tf.bool)

    y = seq2seq.inference(x, t, n_batch, is_training,
                  input_digits=seq2seq.max_len,
                  output_digits=ve.max_len,
                  n_hidden=n_hidden, n_out=n_out)
    loss = seq2seq.loss(y, t)
    train_step = seq2seq.training(loss)

    acc = seq2seq.accuracy(y, t)

    history = {
        'val_loss': [],
        'val_acc': []
    }

    '''
    モデル学習
    '''
    epochs = 200
    batch_size = 10

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    n_batches = N_train // batch_size

    for epoch in range(epochs):
        print('=' * 10)
        print('Epoch:', epoch)
        print('=' * 10)

        X_, Y_ = shuffle(X_train, Y_train)

        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size

            sess.run(train_step, feed_dict={
                x: X_[start:end],
                t: Y_[start:end],
                n_batch: batch_size,
                is_training: True
            })

        # 検証データを用いた評価
        val_loss = loss.eval(session=sess, feed_dict={
            x: X_validation,
            t: Y_validation,
            n_batch: N_validation,
            is_training: False
        })
        val_acc = acc.eval(session=sess, feed_dict={
            x: X_validation,
            t: Y_validation,
            n_batch: N_validation,
            is_training: False
        })

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print('validation loss:', val_loss)
        print('validation acc: ', val_acc)
