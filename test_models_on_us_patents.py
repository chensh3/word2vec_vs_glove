import os
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Embedding, CuDNNLSTM, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from glove import Glove

path = r'C:\Users\Chen\Desktop\Masters_degree\NLP'  # PC Chen
# path = r'C:/Users/Chen/Documents/Masters_degree/word2vec_vs_glove' # Laptop Chen

# trained on only 100 articles
model_embedding = Word2Vec.load(path + r'\output\model_cbow')  ## model_cbow
# model_embedding = Word2Vec.load(path + r'\output\model_sg') ## model_sg
# model_embedding = Glove.load(path + '\output\glove_100.model') ## model_glove

train_df = pd.read_csv(path + r'\us_patent_data\train.csv')
train_df['feature'] = train_df['target'] + ' ' + train_df['anchor']
x = train_df['feature']
y = train_df['score']

# tokenize all words in the train data
token = Tokenizer()
token.fit_on_texts(x)
seq = token.texts_to_sequences(x)
pad_seq = pad_sequences(seq, maxlen=300)
vocab_size = len(token.word_index) + 1

# dictionary of all words and there vectors
embedding_vector = dict(zip(model_embedding.wv.index_to_key, model_embedding.wv.vectors.tolist()))  ## Word2Vec CBoW or SG
# embedding_vector = dict(zip(model_embedding.dictionary,model_embedding.word_vectors.tolist())) ## GloVe


# get matrix of vectors for only words in the train df
embedding_matrix = np.zeros((vocab_size, 300))
for word, i in tqdm(token.word_index.items()):
    embedding_value = embedding_vector.get(word)
    if embedding_value is not None:
        embedding_matrix[i] = embedding_value

model = Sequential()
model.add(Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=300, trainable=False))
model.add(Bidirectional(CuDNNLSTM(75)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='rmsprop')
history = model.fit(pad_seq, y, epochs=50, batch_size=32)

# model.save(path + r'/output/My_Word2vec_LSTM_Model.h5')


from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
token = Tokenizer()
# returns a compiled model
# identical to the previous one
from keras.models import load_model

model = load_model('/kaggle/input/trained-model/My_Word2vec_LSTM_Model.h5')

test_df = pd.read_csv('/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv')
sample = pd.read_csv('/kaggle/input/us-patent-phrase-to-phrase-matching/sample_submission.csv')

# print('sample_shape',sample.shape)
# testing.sample(2)

test_df['feature'] = test_df['target'] + ' ' + test_df['anchor']
x_test = test_df['feature']
token = Tokenizer()
token.fit_on_texts(x_test)
seq = token.texts_to_sequences(x_test)
testing_seq = pad_sequences(seq, maxlen=300)

predict = model.predict(testing_seq)
test_df['label'] = predict

sample['score'] = test_df.label
sample.to_csv("submission1.csv", index=False)
print("Final achieve to send Word2Vec_Predict output data")

#
#
# def load_glove_model(File):
#     print("Loading Glove Model")
#     glove_model = {}
#     with open(File + ".txt",'r',encoding = 'utf-8') as f:
#         for line in f:
#             split_line = line.split()
#             word = split_line[0]
#             embedding = np.array(split_line[1:], dtype=np.float64)
#             glove_model[word] = embedding
#     print(f"{len(glove_model)} words loaded!")
#     return glove_model
#
#
# load_glove_model(r'C:\Users\Chen\Desktop\Masters_degree\NLP\output\glove_100.model')
