import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import word2vec_model
# import glove_model
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords


def data_preprocessing(raw_file_lines):
    data = []
    stop_words = set(stopwords.words('english'))
    # iterate through each sentence in the file
    for i in sent_tokenize(raw_file_lines):
        temp = []
        # tokenize the sentence into words and avoid stop words
        for j in word_tokenize(i):
            if not j.isalpha() or j in stop_words:
                continue

            temp.append(j.lower())
        data.append(temp)

    flat_list = list(set(np.concatenate(data).flat))
    return data, flat_list


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
