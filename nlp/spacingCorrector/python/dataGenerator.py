#!/usr/bin/env python3
# -*- coding: utf8 -*-

import os
import glob
import string
import json
import re
import ast

import jsonlines


MAX_LEN = 128

VOCAB = list(
    string.punctuation +
    string.digits +
    string.ascii_letters +
    ''.join([chr(i) for i in range(ord('가'), ord('힣') + 1)])   
)
stoi = {char: index for index, char in enumerate(['<PAD>']+VOCAB)}
with open('data/stoi.json', 'w', encoding='utf-8') as f:
    json.dump(stoi, f, ensure_ascii=False)
VOCAB = set(VOCAB)

def make_labels(sentence):
    chars = []
    labels = []

    for cnt, char in enumerate(sentence):
        if char == ' ':
            continue
        chars.append(char)
        if cnt + 1 < len(sentence) and sentence[cnt+1] == ' ':
            labels.append(1)
        else:
            labels.append(0)

    return chars, labels

with jsonlines.open('data/final.jsonl', mode='w') as writer:
    for fpath in sorted(glob.glob('data/raw/newspaper/*.json')):
        with open(fpath, 'r', encoding='utf-8') as f:
            raw = json.loads(f.read())

        for document in raw['document']:
            result = []
            for paragraph in document['paragraph'][1:]:
                if paragraph['form'] is not None:
                    sentence = re.sub(r'<.*?>', '', paragraph['form'])
                    source = re.sub(r' ', '', sentence)
                    if (
                        len(source) <= MAX_LEN and
                        all(char in VOCAB for char in source)
                    ):
                        source, label = make_labels(sentence)
                        result.append({'source': source, 'label': label})
            writer.write_all(result)

        print(f"{fpath} Done.")

    for fpath in sorted(glob.glob('data/raw/dialogue/*.json')):
        with open(fpath, 'r', encoding='utf-8') as f:
            raw = json.loads(f.read())

        for document in raw['document']:
            result = []
            for utterance in document['utterance']:
                if utterance['form'] is not None:
                    sentence = utterance['form']
                    source = re.sub(r' ', '', sentence)
                    if (
                        len(source) <= MAX_LEN and
                        all(char in VOCAB for char in source)
                    ):
                        source, label = make_labels(sentence)
                        result.append({'source': source, 'label': label})
            writer.write_all(result)

        print(f"{fpath} Done.")
