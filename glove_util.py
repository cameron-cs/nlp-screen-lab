from gensim.scripts.glove2word2vec import glove2word2vec


def transfer(glove_file, word2vec_file):
    glove2word2vec(glove_file, word2vec_file)
