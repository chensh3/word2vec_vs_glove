from glove import Corpus, Glove


def train_glove(lines, num_comp, num_epoch, model_name=None, lr=0.05):
    # Creating a corpus object
    corpus = Corpus()

    # Training the corpus to generate the co-occurrence matrix which is used in GloVe
    corpus.fit(lines, window=10)

    glove = Glove(no_components=num_comp, learning_rate=lr)
    glove.fit(corpus.matrix, epochs=num_epoch, no_threads=4, verbose=True)
    glove.add_dictionary(corpus.dictionary)
    if model_name != None:
        glove.save(model_name)
    return glove
