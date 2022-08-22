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


# print(df)
full_text = df.full_text
model_cbow, model_sg = [], []
db = []
words = []
for i, patent in enumerate(full_text[:100]):
    print(f"training on file {i}")
    f = patent.replace("\n", " ")
    db_file, words_file = data_preprocessing(f)
    db = db + db_file
    words = words + words_file

flat_list = list(set(words))
model_cbow, model_sg = word2vec_model.train_word2vec(db, 1, 300, 5)
model_cbow.save('C:/Users/Chen/Documents/Masters_degree/word2vec_vs_glove/output/model_cbow')
model_sg.save('C:/Users/Chen/Documents/Masters_degree/word2vec_vs_glove/output/model_sg')
# a = word2vec_model.check_one_word(model_cbow, "network", words)
# b = word2vec_model.check_one_word(model_sg, "network", words)

# get vector of word:
#  model_cbow.wv.get_vector("network") OR model_cbow.wv["network"]
# all weights == all vectors:
# model_cbow.syn1neg

## get model of test from:
## https://www.kaggle.com/code/venkatkumar001/nlp-starter1-almost-all-basic-concept#13.5.-Word-Embedding
## https://www.kaggle.com/code/himanshubag/patent-matching-glove-embedding-lstm
## https://rare-technologies.com/word2vec-tutorial/





# text_path = r"C:\Users\Chen\Documents\Masters_degree\word2vec_vs_glove\alice.txt"
# with open(text_path, encoding='utf-8') as sample:
#     s = sample.read()
#     # Replaces escape character with space
#     f = s.replace("\n", " ")
# db, words = data_preprocessing(f)
# with open('alice1.pickle', 'wb') as handle:
#     pickle.dump(words, handle)

# text_path = r"C:\Users\Chen\Documents\Masters_degree\word2vec_vs_glove\us_patent_data\train.csv"

# db, words = data_preprocessing(f)

# raw_data = pd.read_csv("Salary_Data.csv")

"""
Example of read and use database with word2vec:
"""
# text_path = r"C:\Users\Chen\Documents\Masters_degree\word2vec_vs_glove\alice.txt"
# with open(text_path) as sample:
#     s = sample.read()
#     # Replaces escape character with space
#     f = s.replace("\n", " ")
# db, words = data_preprocessing(f)
#
# model_cbow, model_sg = word2vec_model.train_word2vec(db, 1, 365, 5)
# a = word2vec_model.check_one_word(model_cbow, "alice", words)
# b = word2vec_model.check_one_word(model_sg, "alice", words)
#
# z = pd.DataFrame({"word": words, "cbow": a, "skip_gram": b})
# print(z.head(5))
# z.sort_values("cbow", inplace=True)
# print(z.head(5))
#
# print(model_cbow.wv.most_similar("accounting"))

# mean = z["cbow"].mean()
# below = z[z["cbow"] < mean]
# above = z[z["cbow"] > mean]
# plt.figure(figsize=(20, 12))
# plt.xticks(rotation=90)
# sns.barplot(x="word", y="cbow", data=z,
#             label="Total", color="b")
# sns.displot(z, y="word")
