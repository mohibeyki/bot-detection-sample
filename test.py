import random
import string
from collections import Counter
import pandas as pd
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
import numpy as np

tokenizer = Tokenizer()


def prep_set(file):
    csv_file = pd.read_csv(file + '.csv')
    with open(file + '-dump.csv', 'w') as output:
        for line in csv_file.text.tolist():
            output.write(str(line) + '\n')


def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


def extract_tokens(doc):
    tokens = doc.split()
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


def generate_vocab(bot_tokens, gen_tokens):
    vocab_counter = Counter()
    vocab_counter.update(bot_tokens)
    vocab_counter.update(gen_tokens)
    vocab = {key: val for key, val in vocab_counter.items() if val > 1}
    vocab_list = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    vocab = {text: i + 1 for i, (text, _) in enumerate(vocab_list)}
    return vocab


def get_sequence(tokens, vocab):
    seq = []
    for token in tokens:
        if token in vocab:
            seq.append(vocab[token])
    return seq


def get_dataset(bot_train_tokens, gen_train_tokens, vocab):
    bot_train_seq = get_sequence(bot_train_tokens, vocab)
    gen_train_seq = get_sequence(gen_train_tokens, vocab)
    x_train = bot_train_seq + gen_train_seq
    y_train = [0 for _ in bot_train_seq] + [1 for _ in gen_train_seq]
    train_set = list(zip(x_train, y_train))
    random.shuffle(train_set)
    x, y = zip(*train_set)
    return np.array(x), np.array(y)


def dataset_preparation():
    bot_train_tokens = np.array(extract_tokens(load_doc('tr-bot-dump.csv')))
    gen_train_tokens = np.array(extract_tokens(load_doc('tr-gen-dump.csv')))
    bot_test_tokens = np.array(extract_tokens(load_doc('test-bot-dump.csv')))
    gen_test_tokens = np.array(extract_tokens(load_doc('test-gen-dump.csv')))
    vocab = generate_vocab(bot_train_tokens, gen_train_tokens)
    x_train, y_train = get_dataset(bot_train_tokens, gen_train_tokens, vocab)
    x_test, y_test = get_dataset(bot_test_tokens, gen_test_tokens, vocab)
    return (x_train, y_train), (x_test, y_test)


def main():
    max_features = 20000
    maxlen = 80
    batch_size = 32
    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = dataset_preparation()
    print(x_train, 'train sequences')
    print(x_test, 'test sequences')
    # print('Pad sequences (samples x time)')
    # x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    # x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    # print('x_train shape:', x_train.shape)
    # print('x_test shape:', x_test.shape)
    # print('Build model...')
    # model = Sequential()
    # model.add(Embedding(max_features, 128))
    # model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    # model.add(Dense(1, activation='sigmoid'))
    #
    # # try using different optimizers and different optimizer configs
    # model.compile(loss='binary_crossentropy',
    #               optimizer='adam',
    #               metrics=['accuracy'])
    #
    # print('Train...')
    # model.fit(x_train, y_train,
    #           batch_size=batch_size,
    #           epochs=15,
    #           validation_data=(x_test, y_test))
    # score, acc = model.evaluate(x_test, y_test,
    #                             batch_size=batch_size)
    # print('Test score:', score)
    # print('Test accuracy:', acc)


if __name__ == "__main__":
    # prep_set('tr-bot')
    # prep_set('tr-gen')
    # prep_set('test-bot')
    # prep_set('test-gen')
    main()
