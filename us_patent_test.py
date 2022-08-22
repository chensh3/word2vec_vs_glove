import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import word2vec_model
# import glove_model
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import pickle
from patent_dataset import df
from gensim.models import Word2Vec

# trained on only 100 articles
model_cbow = Word2Vec.load('C:/Users/Chen/Documents/Masters_degree/word2vec_vs_glove/output/model_cbow')
model_sg = Word2Vec.load('C:/Users/Chen/Documents/Masters_degree/word2vec_vs_glove/output/model_sg')

train_df = pd.read_csv(r'C:\Users\Chen\Documents\Masters_degree\word2vec_vs_glove\us_patent_data\train.csv')
print(train_df.info())
# train_df.head()

test_df = pd.read_csv(r'C:\Users\Chen\Documents\Masters_degree\word2vec_vs_glove\us_patent_data\test.csv')
print(test_df.info())
# test_df.head()

from sklearn.preprocessing import LabelEncoder
label_encode = LabelEncoder()

train_df['label'] = label_encode.fit_transform(train_df['score'])

x = train_df['target']
y = train_df['score']

token = Tokenizer()
token.fit_on_texts(x)
seq = token.texts_to_sequences(x)
pad_seq = pad_sequences(seq,maxlen=300)
vocab_size = len(token.word_index)+1

