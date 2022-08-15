import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import word2vec_model
import glove_model


def data_preprocessing(text_path):
    # Reads ‘alice.txt’ file
    with open(text_path) as sample:
        s = sample.read()

    # Replaces escape character with space
    f = s.replace("\n", " ")

    data = []
    stop_words = set(stopwords.words('english'))
    # iterate through each sentence in the file
    for i in sent_tokenize(f):
        temp = []
        # tokenize the sentence into words and avoid stop words
        for j in word_tokenize(i):
            if not j.isalpha() or  j in stop_words:
                continue

            temp.append(j.lower())
        data.append(temp)

    flat_list = list(set(np.concatenate(data).flat))
    return data, flat_list

