import unicodedata
import os
import pandas as pd
import json
import re
import itertools
import torch
from Voca import *


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub('([.!?])', r' \1', s)
    s = re.sub('[^a-zA-Z.!?]+', r' ', s)
    s = re.sub(r'\s+', r' ', s).strip()
    return s


questions = []
answers = []
pairs = []
data_directory = 'dataset/data'
max_length = 30

for data_file in os.listdir(data_directory):
    data = os.path.join(data_directory, data_file)
    data = pd.read_csv(data)
    data = data.dropna()
    questions.extend(data['Question'])
    answers.extend(data['Answer'])

with open('dataset/COVID-QA.json', 'r') as f:
    files = json.load(f)

files = files['data']
for i in range(len(files)):
    file = files[i]
    file = file['paragraphs'][0]

    for j in file['qas']:
        q = j['question']
        a = j['answers'][0]['text']
        questions.append(q)
        answers.append(a)

for i in range(len(questions)):
    q = normalizeString(questions[i])
    a = normalizeString(answers[i])
    pairs.append((q, a))


def filter_pair(p):
    return len(p[0].split(' ')) < max_length and len(p[1].split(' ')) < max_length


def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


def indexFromSentence(voc, s):
    return [voc.word2index[word] for word in s.split(' ')] + [EOS_token]


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


def inputVar(l, voc):
    indexes_batch = [indexFromSentence(voc, s) for s in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)

    return padVar, lengths


def outputVar(l, voc):
    indexes_batch = [indexFromSentence(voc, s) for s in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    mask = binaryMatrix(padList)
    mask = torch.BoolTensor(mask)
    return padVar, mask, max_target_len


def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(' ')), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])

    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len
