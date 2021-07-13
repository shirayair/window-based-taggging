# STUDENT = {'name': "Osnat Ackerman_Shira Yair",
#     'ID': '315747204_315389759'}
import os
import sys

import numpy as np
import load_data
import torch
import torch.nn as nn
from top_k import PreTrainedEmbedding
from tagger1 import train_and_eval_model, create_test, MLP1_model_1, process_data, get_key
from tagger2 import MLP1_model_2, process_data2

HIDDEN_LAYER = 50
EPOCHS = 30
LR = 0.01
EMBEDDING_LENGTH = 50
WINDOW_SIZE = 5


class MLP1_model_3(MLP1_model_1):
    def __init__(self, input_size, hidden_size, output_size, vocab, word_to_idx, labels_to_idx, idx_pre_suf):
        super().__init__(input_size, hidden_size, output_size, vocab, word_to_idx, labels_to_idx)
        self.idx_pre_suf = idx_pre_suf

    def forward(self, x):
        pre_idx, suf_idx = forward_sub(x, self.idx_pre_suf)
        regular = self.embed(x).view(-1, WINDOW_SIZE * EMBEDDING_LENGTH)
        pre_tensor = self.embed(pre_idx).view(-1, WINDOW_SIZE * EMBEDDING_LENGTH)
        suf_tensor = self.embed(suf_idx).view(-1, WINDOW_SIZE * EMBEDDING_LENGTH)
        sum_tensors = regular.add(pre_tensor)
        sum_tensors = sum_tensors.add(suf_tensor)
        hidden = self.fc1(sum_tensors)
        hidden = self.dropout(hidden)
        tanh = self.tanh(hidden)
        output = self.fc2(tanh)
        return self.softmax(output)


class MLP1_model_3_pre_embedding(MLP1_model_2):
    def __init__(self, input_size, hidden_size, output_size, weights_matrix, vocab, word_to_idx, labels_to_idx,
                 idx_pre_suf):
        super().__init__(input_size, hidden_size, output_size, weights_matrix, vocab, word_to_idx, labels_to_idx)
        self.idx_pre_suf = idx_pre_suf

    def forward(self, x):
        pre_idx, suf_idx = forward_sub(x, self.idx_pre_suf)  # getting prefixes and suffixes vectors
        regular = self.embed(x).view(-1, WINDOW_SIZE * EMBEDDING_LENGTH)  # look up embedding for x
        pre_tensor = self.embed(pre_idx).view(-1, WINDOW_SIZE * EMBEDDING_LENGTH)  # look up embedding for prefix
        suf_tensor = self.embed(suf_idx).view(-1, WINDOW_SIZE * EMBEDDING_LENGTH)  # look up embedding for suffix
        sum_tensors = regular.add(pre_tensor)
        sum_tensors = sum_tensors.add(suf_tensor)  # summing the tensors
        hidden = self.fc1(sum_tensors)
        hidden = self.dropout(hidden)
        tanh = self.tanh(hidden)
        output = self.fc2(tanh)
        return self.softmax(output)


def forward_sub(x, idx_pre_suf):
    # creating vectors for suffix and prefix
    pre_idx, suf_idx = x.numpy().copy(), x.numpy().copy()
    pre_idx = pre_idx.reshape(-1)
    suf_idx = suf_idx.reshape(-1)

    # override words indices with their prefixes from the dict idx_pre_suf
    for i, word_idx in enumerate(pre_idx):
        pre_idx[i] = idx_pre_suf[word_idx][0]

    # override words indeces with their suffixes from the dict idx_pre_suf
    for i, word_idx in enumerate(suf_idx):
        suf_idx[i] = idx_pre_suf[word_idx][1]

    # casting back to tensors
    suf_idx = torch.from_numpy(suf_idx.reshape(x.data.shape))
    pre_idx = torch.from_numpy(pre_idx.reshape(x.data.shape))
    pre_idx, suf_idx = pre_idx.type(torch.long), suf_idx.type(torch.long)
    return pre_idx, suf_idx


def tagger3(path, isPos, isSub=True):
    """
    vocab = set of all the words read from train
    word_to_idx = dict of {word: index} - includes all prefixes and suffixes of the words
    labels_to_idx = dict of {label: index}
    idx_pre_suf = dict of indexes. key - word's index. value - tuple (pre_index,suf_index)
    """
    data_loader_train, data_loader_valid, vocab, word_to_idx, labels_to_idx, idx_pre_suf = process_data(path, isPos,
                                                                                                        isSub)
    model = MLP1_model_3(250, HIDDEN_LAYER, len(labels_to_idx), vocab, word_to_idx, labels_to_idx, idx_pre_suf)
    model = model
    # if pos- no weights, if ner - weights= [1.0, 1.0, 1.0, 0.1, 1.0,1.0]
    if not isPos:
        weights = [1.0, 1.0, 1.0, 0.1, 1.0, 1.0]
        class_weights = torch.tensor(weights)
        loss_function = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_function = nn.CrossEntropyLoss()
    model = train_and_eval_model(model, data_loader_train, data_loader_valid, loss_function)
    samples_test, upper_samples = load_data.read_test_data(os.path.join(path, 'test'),
                                                           word_to_idx=word_to_idx,
                                                           labels_to_idx=labels_to_idx)
    create_test('test4.ner', model, samples_test, upper_samples)
    return model


def tagger3_with_pre_embed(path, isPos, isSub=True):
    """
    vocab = set of all the words read from train
    word_to_idx = dict of {word: index} - includes all prefixes and suffixes of the words
    labels_to_idx = dict of {label: index}
    weights_matrix = pre-embedding vectors for all the words including prefixes and suffixes
    idx_pre_suf = dict of indexes. key - word's index. value - tuple (pre_index,suf_index)
    """
    data_loader_train, data_loader_valid, vocab, word_to_idx, labels_to_idx, weights_matrix, idx_pre_suf = process_data2(
        path, isPos, isSub=isSub)
    model = MLP1_model_3_pre_embedding(250, HIDDEN_LAYER, len(labels_to_idx), weights_matrix, vocab, word_to_idx,
                                       labels_to_idx, idx_pre_suf)
    model = model
    # if pos- no weights, if ner - weights= [1.0, 1.0, 1.0, 0.1, 1.0,1.0]
    if not isPos:
        weights = [1.0, 1.0, 1.0, 0.1, 1.0, 1.0]
        class_weights = torch.tensor(weights)
        loss_function = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_function = nn.CrossEntropyLoss()
    model = train_and_eval_model(model, data_loader_train, data_loader_valid, loss_function)
    samples_test, upper_samples = load_data.read_test_data(os.path.join(path, 'test'),
                                                           word_to_idx=word_to_idx, labels_to_idx=labels_to_idx)
    create_test('test4.ner', model, samples_test, upper_samples)
    return model


if __name__ == '__main__':
    args = sys.argv[1:]
    if int(args[2]):
        model = tagger3_with_pre_embed(args[0], int(args[1]), isSub=True)
    else:
        model = tagger3(args[0], int(args[1]), isSub=True)
