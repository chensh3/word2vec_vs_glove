# Python program to generate word vectors using Word2Vec

# importing all necessary modules

from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from nltk.corpus import stopwords
from gensim.models import Word2Vec

import warnings
warnings.filterwarnings(action = 'ignore')



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

def train_word2vec(lines,min_count,vec_size,window_size):

    # Create CBOW model
    model1 = Word2Vec(data, min_count = 1,
                                    vector_size = 365, window = 5)


    # Create Skip Gram model
    model2 = Word2Vec(data, min_count = 1, vector_size = 365,
                                    window = 5, sg = 1)

    return model1,model2


def check_one_word(model,word,flat_list):

    a=[]
    for i in flat_list:
        a.append(model.wv.similarity(word,i))
    return a
#
# z=pd.DataFrame({"word":flat_list,"cbow":a,"skip_gram":b})
# print(z.head(5))
# z.sort_values("cbow",inplace = True)
# print(z.head(5))
# mean=z["cbow"].mean()
# below=z[z["cbow"]<mean]
# above=z[z["cbow"]>mean]
# plt.figure(figsize = (20,12))
# plt.xticks(rotation=90)
# sns.barplot(x="word", y="cbow", data=z,
#             label="Total", color="b")
# # sns.displot(z, y="word")