import os
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Embedding, CuDNNLSTM, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

path = r'C:\Users\Chen\Desktop\Masters_degree\NLP'  # PC Chen
# path = r'C:/Users/Chen/Documents/Masters_degree/word2vec_vs_glove' # Laptop Chen

# trained on only 100 articles
model_cbow = Word2Vec.load(path + r'\output\model_cbow')
model_sg = Word2Vec.load(path + r'\output\model_sg')

train_df = pd.read_csv(path + r'\us_patent_data\train.csv')
x = train_df['target']
y = train_df['score']

# tokenize all words in the train data
token = Tokenizer()
token.fit_on_texts(x)
seq = token.texts_to_sequences(x)
pad_seq = pad_sequences(seq, maxlen = 300)
vocab_size = len(token.word_index) + 1

# dictionary of all words and there vectors
embedding_vector = dict(zip(model_cbow.wv.index_to_key, model_cbow.wv.vectors.tolist()))

# get matrix of vectors for only words in the train df
embedding_matrix = np.zeros((vocab_size, 300))
for word, i in tqdm(token.word_index.items()):
    embedding_value = embedding_vector.get(word)
    if embedding_value is not None:
        embedding_matrix[i] = embedding_value

model = Sequential()
model.add(Embedding(vocab_size, 300, weights = [embedding_matrix], input_length = 300, trainable = False))
model.add(Bidirectional(CuDNNLSTM(75)))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1))
model.compile(loss = 'mse', optimizer = 'rmsprop')
history = model.fit(pad_seq, y, epochs = 50, batch_size = 32)

# model.save(path + r'/output/My_Word2vec_LSTM_Model.h5')


# returns a compiled model
# identical to the previous one
from keras.models import load_model

model = load_model(path + r'/output/My_Word2vec_LSTM_Model.h5')

train_df = pd.read_csv(path + r'\us_patent_data\test.csv')
sample = pd.read_csv(path + r'\us_patent_data\sample_submission.csv')
# print('sample_shape',sample.shape)
# testing.sample(2)

x_test = train_df['target']
x_test = token.texts_to_sequences(x_test)
testing_seq = pad_sequences(x_test, maxlen = 300)

predict = model.predict(testing_seq)
train_df['label'] = predict

sample['score'] = train_df.label
sample.to_csv("submission1.csv", index = False)
print("Final achieve to send Word2Vec_Predict output data")
