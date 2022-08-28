import numpy as np
import pandas as pd
import time
from create_w2v_model import W2V
from nltk.tokenize import sent_tokenize, word_tokenize

pd.options.mode.chained_assignment = None
# from patent_dataset import df
# full_text = df.full_text
def delete_words_not_in_vocab(data, col, model1):
    index = [i for i, target in enumerate(data[col]) if target in model1.vocab]
    filtered_data = data.iloc[index]
    return filtered_data


path = r'C:\Users\Chen\Desktop\Masters_degree\NLP'  # PC Chen

full_text = pd.read_pickle(r"C:\Users\Chen\Desktop\Masters_degree\NLP\full_text.pkl")

model = W2V(300, 2, 1, 0.05, 4)
model.prepare_data(full_text, limit_data = 100)
train_time = model.train_word2vec(num_epochs = 30, model_path = path + r'/output/word2vec_model')

print(f"training the model took : {train_time:0.4f} sec \n")

# model1 = W2V(300, 2, 1, 0.05, 4)
# model1.load_model(r'C:\Users\Chen\Desktop\Masters_degree\NLP\output\word2vec_model') ## NOT WORKING TODO

df = pd.read_csv(path + r'\synonyms.csv').dropna()
df = delete_words_not_in_vocab(df, "lemma", model)
df = df.drop("part_of_speech", axis = 1)
df.synonyms = [i.replace("|", ";").split(";") for i in df.synonyms]

df = df.dropna().reset_index().drop("index", axis = 1)
data = df.groupby("lemma").sum().reset_index().dropna()
data = delete_words_not_in_vocab(data, "lemma", model)


count=[]
for i in range(len(data)):
    count.append(model.check_synonyms_in_model(data.loc[i,"lemma"],data.loc[i,"synonyms"],5))
    print(sum(count))
