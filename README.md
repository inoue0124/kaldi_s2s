# kaldi_s2s
This is a program of a seq2seq language model using Tensorflow and Keras.  
Input : Phoneme posterior vector extracted by Kaldi(A toolkit of automatic speech recognition).  
Output : Intended text  

## Description
ASR (Automatic speech recognition) is one of the hottest study area, and many kind of approach has been proposed for a long time.  
A basic model of ASR is a combination of acostic model and language model.   
This program is a language model using seq2seq encoder-decoder, which converts a frame sequence of phoneme posterior vector into an intended text.  


## Usage
### Install
Clone this repository to your computer.

```sh
git clone https://github.com/inoue0124/kaldi_s2s.git
cd kaldi_s2s
```
###  Preparation of training data
It is necessary to place the phoneme posterior vector data in text format in the data directory.  
Please calculate phoneme posterior by decoding speech files with kaldi.  
It is assumed that one utterance is stored in one file.  

### Run
At first, rewirte the number of data, epoch, batch size (Written in keras_seq2seq.py)  
Then, execute a below command,

```sh
python keras_seq2seq.py {data_name} train
```

A log directory will be made and various files will be created in it.  
if you want to check a learning history on tensorboard, do the command like
```sh
tensorboard --logdir=./log/{time_stamp}
```
and access here [http://127.0.0.1:6006/](http://127.0.0.1:6006/).  

### Prediction with learned weights
After learning, you can test the predicition by doing this
```sh
python keras_seq2seq.py {data_name} test {log_dir_name}
```
Here, log_dir_name has been named when learning phase.


## Reference
* [Kaldi](https://github.com/kaldi-asr/kaldi)
* [Seq2seq](https://github.com/udacity/deep-learning/blob/master/seq2seq/sequence_to_sequence_implementation.ipynb)
* [詳解 ディープラーニング](https://github.com/yusugomori/deeplearning-tensorflow-keras/tree/r1.4)

