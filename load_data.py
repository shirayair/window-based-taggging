# STUDENT = {'name': "Osnat Ackerman_Shira Yair",
#     'ID': '315747204_315389759'}
import torch
import torch.nn as nn
import re
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import copy

BATCH_SIZE = 100


def read_test_data(path, word_to_idx=None, labels_to_idx=None):
    origin_samples = []

    all_samples = []
    with open(path, 'r') as freader:
        row = freader.readline().rstrip('\n')
        while row:
            rows = []
            while row != '':
                origin_samples.append(row)
                row = row.lower()
                row = re.sub('[0-9]', 'DG', row)
                rows.append(row)
                row = freader.readline().rstrip('\n')
            samples = create_input(rows)
            all_samples.extend(samples)
            row = freader.readline().rstrip('\n')

        vocab, all_samples, all_labels, word_to_idx, labels_to_idx = convert_word_to_idx(all_samples, None, None,
                                                                                       word_to_idx,
                                                                                       labels_to_idx)
        return all_samples, origin_samples


def read_data(path, isPos=True, word_to_idx=None, labels_to_idx=None, isSub=False):
    idx_pre_suf = {}
    vocab = set()
    all_sampels, all_labels = [], []
    with open(path, 'r') as freader:
        row = freader.readline().rstrip('\n')
        while row:
            rows = []
            while row != '':
                rows.append(row)
                row = freader.readline().rstrip('\n')
            samples, labels = split_to_samples_lables(rows, isPos)
            vocab.update(samples)
            features = create_input(samples)
            all_sampels.extend(features)
            all_labels.extend(labels)
            row = freader.readline().rstrip('\n')
    vocab.add('<START>')
    vocab.add('<END>')
    if (isSub):
        vocab, all_samples, all_labels, word_to_idx, labels_to_idx, idx_pre_suf = convert_sub_word_to_idx(all_sampels,
                                                                                                        all_labels,
                                                                                                        vocab,
                                                                                                        word_to_idx,
                                                                                                        labels_to_idx)
    else:
        vocab, all_samples, all_labels, word_to_idx, labels_to_idx = convert_word_to_idx(all_sampels, all_labels, vocab,
                                                                                       word_to_idx,
                                                                                       labels_to_idx)
    return vocab, all_samples, all_labels, word_to_idx, labels_to_idx, idx_pre_suf


def split_to_samples_lables(rows, isPos):
    delim = ' ' if isPos else '\t'
    samples, labels = [], []
    for r in rows:
        split = r.split(delim)
        word = split[0]
        word = word.lower()
        word = re.sub('[0-9]', 'DG', word)
        samples.append(word)
        labels.append(split[1])
    return samples, labels


def create_input(samples):
    samples.insert(0, '<START>')
    samples.insert(1, '<START>')
    samples.append('<END>')
    samples.append('<END>')

    input = []
    for i in range(2, len(samples) - 2):
        vector = samples[i - 2:i + 3]
        input.append(vector)

    return input


def convert_sub_word_to_idx(samples, labels, vocab, word_to_idx, labels_to_idx):
    all_samples, all_labels = [], []
    word_to_pre_suf = {}
    idx_to_pre_suf = {}
    if not word_to_idx:
        vocab.add('')
        sub_vocab = set()
        for word in vocab:
            pre_word, suf_word = word[:3], word[-3:]
            sub_vocab.add(suf_word)
            sub_vocab.add(pre_word)

        sub_vocab = vocab.union(sub_vocab)
        sub_vocab = list(sorted(sub_vocab))
        word_to_idx = {word: i for i, word in enumerate(sub_vocab)}
        for word, idx in word_to_idx.items():
            idx_to_pre_suf[idx] = (word_to_idx[word[:3]], word_to_idx[word[-3:]])
        vocab = set(sub_vocab)
        vocab_l = set(labels)
        vocab_l.add('')
        vocab_l = list(sorted(vocab_l))
        labels_to_idx = {word: i for i, word in enumerate(vocab_l)}
        all_samples = []
        for sample in samples:
            sample = [word_to_idx[word] for word in sample]
            all_samples.append(sample)
        all_labels = [labels_to_idx[word] for word in labels]
    else:
        for sample in samples:
            sample = [word_to_idx[word] if word in word_to_idx else word_to_idx[''] for word in sample]
            all_samples.append(sample)
        if labels:
            all_labels = [labels_to_idx[word] if word in labels_to_idx else labels_to_idx[''] for word in labels]

    return vocab, all_samples, all_labels, word_to_idx, labels_to_idx, idx_to_pre_suf


def convert_word_to_idx(samples, labels, vocab, word_to_idx, labels_to_idx):
    all_samples, all_labels = [], []
    if not word_to_idx:
        vocab.add('')
        vocab = list(sorted(vocab))
        word_to_idx = {word: i for i, word in enumerate(vocab)}
        vocab_l = set(labels)
        vocab_l.add('')
        vocab_l = list(sorted(vocab_l))
        labels_to_idx = {word: i for i, word in enumerate(vocab_l)}
        all_samples = []
        for sample in samples:
            sample = [word_to_idx[word] for word in sample]
            all_samples.append(sample)
        all_labels = [labels_to_idx[word] for word in labels]
    else:
        for sample in samples:
            sample = [word_to_idx[word] if word in word_to_idx else word_to_idx[''] for word in sample]
            all_samples.append(sample)
        if labels:
            all_labels = [labels_to_idx[word] if word in labels_to_idx else labels_to_idx[''] for word in labels]

    return vocab, all_samples, all_labels, word_to_idx, labels_to_idx


def make_loader(samples, labels):
    x, y = samples, labels
    x, y = torch.from_numpy(np.array(x)), torch.from_numpy(np.array(y))
    x, y = x.type(torch.long), y.type(torch.long)
    return DataLoader(TensorDataset(x, y), BATCH_SIZE, shuffle=True)


def make_test_loader(samples):
    x = samples
    x = torch.from_numpy(np.array(x))
    x = x.type(torch.long)
    return DataLoader(TensorDataset(x), 1, shuffle=False)


if __name__ == '__main__':
    vocab, all_samples, all_labels, word_to_idx, labels_to_idx, idx_pre_suf = read_data("./pos/train", isSub=True)
    print(idx_pre_suf)

    # p = make_loader(all_samples, all_labels)
    # _samples = read_test_data("./pos/test", word_to_idx, labels_to_idx)
    # print(_samples)
