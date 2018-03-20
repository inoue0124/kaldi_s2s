# kaldi_s2s
This is a program of a seq2seq language model using Tensorflow and Keras.
Input : Phoneme posterior vector extracted by Kaldi(A toolkit of automatic speech recognition).
Output : Intended text

## Description


## Usage
### Install
At first, clone this repository to your computer.

```sh
git clone https://github.com/inoue0124/kaldi_s2s.git
cd kaldi_s2s
```

Then, rewirte the number of data, epoch, batch size (Written in keras_seq2seq.py)

After that, execute a below command,

```sh
python keras_seq2seq.py
```

A log directory will be made and various files will be created in it.
if you want to check a learning history on tensorboard, do the command like

```sh
tensorboard --logdir=./log/{time_stamp}
```
and access here [http://127.0.0.1:6006/](http://127.0.0.1:6006/)


### Open
特定のソフトで開くとき

### Settings
環境変数など

### Build
コンパイルなど

### Deploy
![herokubutton](https://www.herokucdn.com/deploy/button.svg)  
とかでもいい

### Run
実行

### Check
```sh
firefox http://localhost:8080/my-project &
```
とか

## Hints
### Options
コマンドのオプションとか

### Distribute
`*.zip`にするタスクとかあるなら

### Examples Of Command
コマンドの実行結果の例とか

## Future Releases
今後の方針

## Contribution
1. Fork it  
2. Create your feature branch  
3. Commit your changes  
4. Push to the branch  
5. Create new Pull Request 
