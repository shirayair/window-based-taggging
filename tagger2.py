# STUDENT = {'name': "Osnat Ackerman_Shira Yair",
#     'ID': '315747204_315389759'}
import os
import re
import sys

import numpy as np
import load_data
import torch
import torch.nn as nn
from top_k import PreTrainedEmbedding
from tagger1 import train_and_eval_model, create_test

HIDDEN_LAYER = 50
EPOCHS = 20
LR = 0.01
EMBEDDING_LENGTH = 50
WINDOW_SIZE = 5


class MLP1_model_2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, weights_matrix, vocab, word_to_idx, labels_to_idx):
        super(MLP1_model_2, self).__init__()
        torch.manual_seed(3)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        weights_matrix = torch.tensor(weights_matrix)
        weights_matrix = weights_matrix.type(torch.float32)
        self.embed = nn.Embedding.from_pretrained(weights_matrix, freeze=False)
        self.word_to_idx = word_to_idx
        self.labels_to_idx = labels_to_idx
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.embed(x).view(-1, EMBEDDING_LENGTH * WINDOW_SIZE)
        hidden = self.fc1(x)
        hidden = self.dropout(hidden)
        tanh = self.tanh(hidden)
        output = self.fc2(tanh)
        return self.softmax(output)


def create_weghits(word_to_idx, isSub):
    pre_embedding = PreTrainedEmbedding()
    yoav_vocab = pre_embedding.get_vocab()
    yoav_vecs = pre_embedding.get_vecs()
    yoav_W2I = pre_embedding.get_W2I()

    new_index = len(word_to_idx)
    for word in yoav_vocab:
        if word not in word_to_idx.keys():
            word_to_idx[word] = new_index
            if isSub:
                prefix = word[:3]
                sufix = word[-3:]
                if prefix not in word_to_idx.keys():
                    word_to_idx[prefix] = new_index + 1
                    new_index += 1
                if sufix not in word_to_idx.keys():
                    word_to_idx[sufix] = new_index + 1
                    new_index += 1
            new_index += 1

    matrix_len = len(word_to_idx)
    weights_matrix = np.zeros((matrix_len, 50))

    for word, index in word_to_idx.items():
        try:
            weights_matrix[index] = yoav_vecs[yoav_W2I[word]]
            yoav_vocab.remove(word)
        except KeyError:
            weights_matrix[index] = np.random.normal(scale=0.6, size=(1, EMBEDDING_LENGTH))

    return word_to_idx, weights_matrix


def process_data2(path, isPos, isSub=False):
    vocab, all_samples, all_labels, word_to_idx, labels_to_idx, idx_pre_suf = load_data.read_data(
        os.path.join(path, 'train'), isPos=isPos, isSub=isSub)
    data_loader_train = load_data.make_loader(all_samples, all_labels)

    word_to_idx, weights_matrix = create_weghits(word_to_idx, isSub)

    if isSub:
        for word, ix in word_to_idx.items():
            if ix not in idx_pre_suf.keys():
                idx_pre_suf[ix] = (word_to_idx[word[:3]], word_to_idx[word[-3:]])

    _, all_samples_val, all_labels_val, _, _, _ = load_data.read_data(os.path.join(path, 'dev'),
                                                                      isPos=isPos, word_to_idx=word_to_idx,
                                                                      labels_to_idx=labels_to_idx, isSub=isSub)

    data_loader_valid = load_data.make_loader(all_samples_val, all_labels_val)
    return data_loader_train, data_loader_valid, vocab, word_to_idx, labels_to_idx, weights_matrix, idx_pre_suf


def tagger2(path, isPos, isSub=False):
    data_loader_train, data_loader_valid, vocab, word_to_idx, labels_to_idx, weights_matrix, idx_pre_suf = process_data2(
        path, isPos, isSub)
    model = MLP1_model_2(250, HIDDEN_LAYER, len(labels_to_idx), weights_matrix, vocab, word_to_idx, labels_to_idx)
    model = model
    if not isPos:
        # if pos- no weights, if ner - weights= [1.0, 1.0, 1.0, 0.1, 1.0,1.0]
        weights = [1.0, 1.0, 1.0, 0.1, 1.0, 1.0]
        class_weights = torch.tensor(weights)
        loss_function = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_function = nn.CrossEntropyLoss()
    model = train_and_eval_model(model, data_loader_train, data_loader_valid, loss_function)
    samples_test, upper_samples = load_data.read_test_data(os.path.join(path, 'test'), word_to_idx, labels_to_idx)
    create_test('test3.pos', model, samples_test, upper_samples)
    return model


if __name__ == '__main__':
    args = sys.argv[1:]
    model = tagger2(args[0], int(args[1]))
