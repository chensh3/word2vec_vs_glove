# from glove import  Glove
# # from glove import Corpus
#
#
#
#
#
import numpy as np
import pandas as pd
#
def create_cooccurrence_matrix(sentences, window_size=2):
    """Create co occurrence matrix from given list of sentences.

    Returns:
    - vocabs: dictionary of word counts
    - co_occ_matrix_sparse: sparse co occurrence matrix

    Example:
    ===========
    sentences = ['I love nlp',    'I love to learn',
                 'nlp is future', 'nlp is cool']

    vocabs,co_occ = create_cooccurrence_matrix(sentences)

    df_co_occ  = pd.DataFrame(co_occ.todense(),
                              index=vocabs.keys(),
                              columns = vocabs.keys())

    df_co_occ = df_co_occ.sort_index()[sorted(vocabs.keys())]

    df_co_occ.style.applymap(lambda x: 'color: red' if x>0 else '')

    """
    import scipy
    import nltk

    vocabulary = {}
    data = []
    row = []
    col = []

    tokenizer = nltk.tokenize.word_tokenize

    for sentence in sentences:
        # sentence = sentence.strip()
        # print( sentence)
        tokens = [token for token in tokenizer(sentence) if token != u""]
        # print(tokens)
        for pos, token in enumerate(tokens):
            i = vocabulary.setdefault(token, len(vocabulary))
            start = max(0, pos-window_size)
            end = min(len(tokens), pos+window_size+1)
            for pos2 in range(start, end):
                if pos2 == pos:
                    continue
                j = vocabulary.setdefault(tokens[pos2], len(vocabulary))
                data.append(1.)
                row.append(i)
                col.append(j)

    cooccurrence_matrix_sparse = scipy.sparse.coo_matrix((data, (row, col)))
    return vocabulary, cooccurrence_matrix_sparse
#
#
# # importing required libraries
#
# import gensim
#
# from gensim import corpora
#
# # creating a sample corpus for demonstration purpose
#
txt_corpus = [

    "Find end to end projects at ProjectPro",

    # "Stop wasting time on different online forums to get your project solutions",
    #
    # "Each of our projects solve a real business problem from start to finish",
    #
    # "All projects come with downloadable solution code and explanatory videos",

    "All our projects are designed modularly so you can rapidly learn and reuse modules"]
#
# # Creating a set of frequent words
#
stoplist = set('for a of the and to in on of to are at'.split(' '))

# Lowercasing each document, using white space as delimiter and filtering out the stopwords

processed_text = [[word for word in document.lower().split() if word not in stoplist] for document in txt_corpus]

# # creating a dictionary
#
# dictionary = corpora.Dictionary(processed_text)
#
# # displaying the dictionary
#
# print(dictionary)
#
# d, mat=create_cooccurrence_matrix(processed_text)
# def train_glove(lines, num_comp, num_epoch,dic, model_name=None, lr=0.05):
#     # Creating a corpus object
#     # corpus = Corpus()
#
#     # Training the corpus to generate the co-occurrence matrix which is used in GloVe
#     # corpus.fit(lines, window=10)
#
#     glove = Glove(no_components=num_comp, learning_rate=lr)
#     glove.fit(mat, epochs=num_epoch, no_threads=4, verbose=True)
#     glove.add_dictionary(dic)
#     if model_name != None:
#         glove.save(model_name)
#     return glove
#
# g=train_glove(processed_text,5,30,dictionary)
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
def data_preprocessing(raw_file):
    data = []
    stop_words = set(stopwords.words('english'))
    # iterate through each sentence in the file
    for i in sent_tokenize(raw_file):
        temp = []
        # tokenize the sentence into words and avoid stop words
        for j in word_tokenize(i):
            if not j.isalpha():
                # print(j)
                # print(f"not a word: {j}")
                continue

            temp.append(j.lower())
        data.append(temp)

    flat_list = list(set(np.concatenate(data).flat))
    return data, flat_list


# path = r'C:\Users\Chen\Desktop\Masters_degree\NLP' # PC Chen
# # path = r'C:/Users/Chen/Documents/Masters_degree/word2vec_vs_glove' # Laptop Chen
#
#
# # from glove import Corpus, Glove
#
# text_path = path + r'\alice.txt'
# with open(text_path, encoding='utf-8') as sample:
#     s = sample.read()
#     # Replaces escape character with space
#     f = s.replace("\n", " ")
# db, words = data_preprocessing(txt_corpus)
# #
#
# #data
# lines = [['woman', 'umbrella', 'silhouetted'], ['person', 'black', 'umbrella']]
# d, mat=create_cooccurrence_matrix(txt_corpus)
#
from glove import *
#
# model = Glove(mat.toarray().astype(int))
# model.train(db)
# model.to_txt()
# words = model.most_similary("projects", 10)
# print(words)

text_path = r"C:\Users\Chen\Desktop\Masters_degree\NLP\glove_py\data\small_text2"
with open(text_path, encoding='utf-8') as sample:
    s = sample.read()
    # Replaces escape character with space
    f = s.replace("\n", " ")
db, words = data_preprocessing(f)
flat_list = list(set(words))
x=np.zeros(len(flat_list))

d, mat=create_cooccurrence_matrix(sent_tokenize(f))
matx=mat.toarray().astype("int64")
dictory=dict(zip(words, matx))
model = Glove(matx)

model.train(dictory)
#
#
# # model.load("log/vocab.txt", "log/wordvec.txt", "log/wordvec.txt.meta")
# words = model.most_similary("one", 10)
# print(words)

#importing required libraries

import gensim

from gensim import corpora

from gensim.utils import simple_preprocess

from smart_open import smart_open

import os



#creating a dictionary from a single text file

gensim_dictionary = corpora.Dictionary(simple_preprocess(line, deacc =False) for line in open(text_path, encoding='utf-8'))

print("The dictionary has: " +str(len(gensim_dictionary)) + " tokens")
dicty={}

for k, v in gensim_dictionary.token2id.items():

     print(f'{k:{15}} {v:{10}}')
     dicty[k]=v