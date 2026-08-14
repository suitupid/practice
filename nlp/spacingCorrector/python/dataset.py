#!/usr/bin/python3
# -*- coding: utf8 -*-

import joblib
import json

import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):

    def __init__(self, datasets):
        self.datasets = datasets
        self.max_len = 128
        with open('data/stoi.json', 'r', encoding='utf-8') as f:
            self.stoi = json.load(f)

    def _open_file(self, fpath):
        self.f = open(fpath, 'rb')

    def encode_source(self, text):
        text = text[:self.max_len]
        indexes = [self.stoi[char] for char in text]
        indexes += [self.stoi['<PAD>']] * (self.max_len - len(indexes))
        return indexes

    def encode_label(self, label):
        label = label[:self.max_len]
        label += [-100] * (self.max_len - len(label))
        return label

    def __getitem__(self, index):
        offset = self.datasets[index]
        self._open_file('data/final.jsonl')
        self.f.seek(offset)
        item = self.f.readline().decode('utf-8').strip()
        item = json.loads(item)
        x = self.encode_source(item['source'])
        y = self.encode_label(item['label'])
        x = torch.tensor(x)
        y = torch.tensor(y)
        return x, y

    def __len__(self):
        return len(self.datasets)
