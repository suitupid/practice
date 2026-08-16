#!/usr/bin/env python3
# -*- coding: utf8 -*-

import json

import torch

from model import CustomNetwork


torch.backends.mkldnn.enabled = False

class Inference():

    def __init__(self):
        self.model = CustomNetwork()
        self.model.load_state_dict(torch.load('model/spacingCorrector.pt'))
        self.model.eval()
        with open('data/stoi.json', 'r', encoding='utf-8') as f:
            self.stoi = json.load(f)
        self.pad_idx = self.stoi['<PAD>']
        self.max_len = 128

    def preprocess(self, text):
        text = text[:self.max_len]
        ids = [self.stoi.get(char, self.pad_idx) for char in text]
        ids += [self.pad_idx] * (self.max_len - len(ids))
        return torch.tensor(ids).unsqueeze(0)

    def predict(self, text):
        text = text.replace(' ', '')
        x = self.preprocess(text)

        with torch.no_grad():
            pred = torch.argmax(self.model(x), dim=-1)
        pred = pred.squeeze(0).cpu().numpy()
        pred = pred[:len(text)]

        result = []
        for char, tag in zip(text, pred):
            result.append(char)
            if tag == 1:
                result.append(' ')
        result = ''.join(result).strip()

        return result