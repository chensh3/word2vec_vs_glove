# Python program to generate word vectors using Word2Vec

# importing all necessary modules
from gensim.models import Word2Vec
import warnings

warnings.filterwarnings(action='ignore')


def train_word2vec(lines, min_count, vec_size, window_size):
    print("\ntraining new models")
    # Create CBOW model
    model1 = Word2Vec(lines, min_count=min_count,
                      vector_size=vec_size, window=window_size)

    # Create Skip Gram model
    model2 = Word2Vec(lines, min_count=min_count, vector_size=vec_size,
                      window=window_size, sg=1)
    print("\nFinished training new models")
    return model1, model2


def check_one_word(model, word, flat_list):
    a = []
    for i in flat_list:
        a.append(model.wv.similarity(word, i))
    return a
#

# # Print results
# print("Cosine similarity between 'alice' " +
#       "and 'wonderland' - CBOW : ",
#       model1.wv.similarity('alice', 'wonderland'))
#
# print("Cosine similarity between 'alice' " +
#       "and 'machines' - CBOW : ",
#       model1.wv.similarity('alice', 'machines'))
#


# # Print results
# print("Cosine similarity between 'alice' " +
#       "and 'wonderland' - Skip Gram : ",
#       model2.wv.similarity('alice', 'lewis'))
#
# print("Cosine similarity between 'alice' " +
#       "and 'machines' - Skip Gram : ",
#       model2.wv.similarity('alice', 'day'))
