# STUDENT = {'name': "Osnat Ackerman_Shira Yair",
#     'ID': '315747204_315389759'}

import os
import sys

import load_data
import torch
import torch.nn as nn

HIDDEN_LAYER = 50
EPOCHS = 30
LR = 0.01
EMBEDDING_LENGTH = 50
WINDOW_SIZE = 5


class MLP1_model_1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, vocab, word_to_idx, labels_to_idx):
        super(MLP1_model_1, self).__init__()
        torch.manual_seed(3)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embed = nn.Embedding(len(word_to_idx), EMBEDDING_LENGTH)
        nn.init.uniform_(self.embed.weight, -1.0, 1.0)
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


def get_key(val, labels_to_idx):
    for key, value in labels_to_idx.items():
        if val == value:
            return key
    return None


def calc_accuracy(predict, y_label, labels_to_idx):
    """
     counting good predictions and divide by number of predictions.
        for NER data, counting only success that are not 'O' label
    """
    dom = len(y_label)
    good = 0
    for p, l in zip(predict, y_label):
        if p.argmax() == l:
            if not get_key(int(l), labels_to_idx) == 'O':
                good += 1
            else:
                dom -= 1
    if dom == 0:
        return 0
    return good / float(dom)


def validation_check(i, model, valid_loader, loss_func):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    print(f'Epoch: {i + 1:02} | Starting Evaluation...')
    for x, y_label in valid_loader:
        predict = model(x)
        loss = loss_func(predict, y_label)
        epoch_acc += calc_accuracy(predict, y_label, model.labels_to_idx)
        epoch_loss += loss
    print(f'Epoch: {i + 1:02} | Finished Evaluation')
    return float(epoch_loss) / len(valid_loader), float(epoch_acc) / len(valid_loader)


def train_and_eval_model(model, data_loader, valid_loader, loss_func):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    for i in range(EPOCHS):
        epoch_loss = 0
        epoch_acc = 0
        model.train()
        print(f'Epoch: {i + 1:02} | Starting Training...')
        for x, y in data_loader:
            optimizer.zero_grad()
            prediction = model(x)
            loss = loss_func(prediction, y)
            epoch_acc += calc_accuracy(prediction, y, model.labels_to_idx)
            epoch_loss += loss
            loss.backward()
            optimizer.step()
        print(f'Epoch: {i + 1:02} | Finished Training')
        avg_epoch_loss, avg_epoch_acc = float(epoch_loss) / len(data_loader), float(epoch_acc) / len(data_loader)
        avg_epoch_loss_val, avg_epoch_acc_val = validation_check(i, model, valid_loader, loss_func)
        print(f'\tTrain Loss: {avg_epoch_loss:.3f} | Train Acc: {avg_epoch_acc * 100:.2f}%')
        print(f'\t Val. Loss: {avg_epoch_loss_val:.3f} |  Val. Acc: {avg_epoch_acc_val * 100:.2f}%')
    return model


def create_test(path, model, samples, origin_samples):
    model.eval()
    test_loader = load_data.make_test_loader(samples)
    with open(path, 'w') as fwriter:
        for i, x in enumerate(test_loader):
            x = x[0]  # extracting value from tensor
            predict = model(x)
            p = predict.argmax()
            x = x.data.tolist()
            label = get_key(p, model.labels_to_idx)
            row = origin_samples[i] + ' ' + label + '\n'  # back to upper case
            fwriter.write(row)
            if x[0][3] == model.word_to_idx["<END>"]:  # when reaching end of sentence- empty line
                fwriter.write('\n')
    fwriter.close()


def process_data(path, isPos, isSub=False):
    vocab, all_samples, all_labels, word_to_idx, labels_to_idx, idx_pre_suf = load_data.read_data(
        os.path.join(path, 'train'),
        isPos=isPos, isSub=isSub)
    data_loader_train = load_data.make_loader(all_samples, all_labels)
    _, all_samples_val, all_labels_val, _, _, _ = load_data.read_data(os.path.join(path, 'dev'),
                                                                      isPos=isPos, word_to_idx=word_to_idx,
                                                                      labels_to_idx=labels_to_idx, isSub=isSub)
    data_loader_valid = load_data.make_loader(all_samples_val, all_labels_val)
    return data_loader_train, data_loader_valid, vocab, word_to_idx, labels_to_idx, idx_pre_suf


def tagger1(path, isPos):
    data_loader_train, data_loader_valid, vocab, word_to_idx, labels_to_idx, idx_pre_suf = process_data(path, isPos)
    model = MLP1_model_1(250, HIDDEN_LAYER, len(labels_to_idx), vocab, word_to_idx, labels_to_idx)
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
    create_test('test1.ner', model, samples_test, upper_samples)
    return model


if __name__ == '__main__':
    args = sys.argv[1:]
    model = tagger1(args[0], int(args[1]))
