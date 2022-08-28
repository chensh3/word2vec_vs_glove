import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
import time

path = r'C:\Users\Chen\Desktop\Masters_degree\NLP'  # PC Chen


# path = r'C:/Users/Chen/Documents/Masters_degree/word2vec_vs_glove' # Laptop Chen
class W2V():
    def __init__(self, vector_size, window_size, sg = 1, lr = 0.05, workers = 4):
        self.vector_size = vector_size
        self.window_size = window_size
        self.sg = sg
        self.learning_rate = lr
        self.workers = workers

        self.data = None
        self.model = None
        self.vocab = None

    def train_word2vec(self, num_epochs, model_path = ""):
        print("\n\nTraining word2vec model")
        # Create Skip Gram model ( sg=1 )  or CBOW ( sg=0 )
        start = time.perf_counter()

        word2vec = Word2Vec(self.data, vector_size = self.vector_size, window = self.window_size,
                            sg = self.sg, epochs = num_epochs, workers = self.workers, alpha = self.learning_rate)

        end = time.perf_counter()

        print("Finished training word2vec model\n")

        self.model = word2vec
        if model_path != "":
            self.model.save(model_path)

        return end - start

    def check_one_word(self, word):
        a = []
        for i in self.vocab:
            a.append(self.model.wv.similarity(word, i))
        return a

    @staticmethod
    def data_preprocessing(raw_file):
        data = []
        # iterate through each sentence in the file
        for i in sent_tokenize(raw_file):
            temp = []
            # tokenize the sentence into words and avoid stop words
            for j in word_tokenize(i):
                if not j.isalpha():
                    continue

                temp.append(j.lower())
            data.append(temp)

        flat_list = list(set(np.concatenate(data).flat))
        return data, flat_list

    def prepare_data(self, df, limit_data = None):
        db = []
        words = []
        for i, patent in enumerate(df[:limit_data]):
            print(f"training on patent num: {i}", end = '\r')
            f = patent.replace("\n", " ")
            db_file, words_file = self.data_preprocessing(f)
            db = db + db_file
            words = words + words_file

        flat_list = list(set(words))

        self.data = db
        self.vocab = flat_list

    def delete_unknown_words(self, words):

        known_words = [word for word in words if word in self.vocab]
        unknown_words = list(set(words) - set(known_words))
        return known_words, unknown_words

    def get_top_similar_words(self, word, num = 5):
        most_similar = np.array(self.model.wv.most_similar(word, topn = num)).transpose()[0]
        return most_similar

    @staticmethod
    def check_synonyms_in_model(target, words, num):
        count = 0
        similar_words = get_top_similar_words(target, num)
        for word in words:
            if word in similar_words:
                count += 1
        return count


from patent_dataset import df

full_text = df.full_text

model = W2V(300, 2, 1, 0.05, 4)
model.prepare_data(full_text, limit_data = 100)
train_time = model.train_word2vec(num_epochs = 30, model_path = path + r'/output/word2vec_model')

print(f"training the model took : {train_time:0.4f} sec \n")

# get vector of word:
#  model_cbow.wv.get_vector("network") OR model_cbow.wv["network"]

# all weights == all vectors:
# model_cbow.syn1neg

## get words vocab
# model_cbow.wv.index_to_key


## get model of test from:
## https://www.kaggle.com/code/venkatkumar001/nlp-starter1-almost-all-basic-concept#13.5.-Word-Embedding
## https://www.kaggle.com/code/himanshubag/patent-matching-glove-embedding-lstm
## https://rare-technologies.com/word2vec-tutorial/
