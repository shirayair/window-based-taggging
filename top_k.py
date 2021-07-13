# STUDENT = {'name': "Osnat Ackerman_Shira Yair",
#     'ID': '315747204_315389759'}
import numpy as np
import operator as op


class PreTrainedEmbedding:
    def __init__(self):
        self.vocab = self.read_vocab("vocab.txt")
        self.vecs = np.loadtxt("wordVectors.txt")
        self.W2I = self.word_to_idx()

    def read_vocab(self, path):
        vocab = []
        with open(path, 'r') as freader:
            row = freader.readline().rstrip('\n')
            while row:
                vocab.append(row)
                row = freader.readline().rstrip('\n')
        freader.close()
        return vocab

    def word_to_idx(self):
        word_to_idx = {v: i for i, v in enumerate(self.vocab)}
        return word_to_idx

    def get_vocab(self):
        return self.vocab

    def get_vecs(self):
        return self.vecs

    def get_W2I(self):
        return self.W2I


def most_similar(word, k):
    word_to_distance = {}
    embedding = PreTrainedEmbedding()
    u = embedding.vecs[embedding.W2I[word]]
    for w in embedding.vocab:
        if w == word:
            continue
        v = embedding.vecs[embedding.W2I[w]]
        distance = sim(u, v)
        word_to_distance[w] = distance
    sorted_dic = sorted(word_to_distance.items(), key=op.itemgetter(1), reverse=True)
    return sorted_dic[:k]


def sim(u, v):
    a = np.sqrt(np.dot(u, u))
    b = np.sqrt(np.dot(v, v))
    numerator = np.dot(u, v)
    denominator = np.dot(a, b)
    return numerator / denominator


if __name__ == '__main__':
    list_word = ["dog", "england", "john", "explode", "office"]
    for word in list_word:
        most_sim = most_similar(word, 5)
        print(most_sim)
