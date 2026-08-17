#!/usr/bin/env python3
# -*- coding: utf8 -*-

import os
import json
import joblib
from pathlib import Path
from multiprocessing import Process, Pool

import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


RAW_PATH = Path('data/raw')
IMAGE_PATH = RAW_PATH / 'image'
LABEL_PATH = RAW_PATH / 'label'
SAVE_PATH = Path('data/final')

with open('data/targetSyllable.txt', 'r') as f:
    target = f.read().replace('\n', '')
encoder = LabelEncoder().fit(list(target))
with open('data/labelEncoder.bin','wb') as f:
    joblib.dump(encoder, f)

label_paths = [str(path) for path in LABEL_PATH.rglob('*.json')]

print('Find Target..')
data_info = []
for label_path in label_paths:
    with open(label_path, 'r') as f:
        data_info_raw = json.loads(f.read())
    label = data_info_raw['text']['letter']['value']
    if label in target:
        image_name = data_info_raw['image']['file_name']
        image_path = IMAGE_PATH / image_name[:3]
        data_info.append([image_path, image_name, label])
print('Find Target Done.')

os.mkdir('data/final') if not os.path.isdir('data/final') else False

print('Create Train Images..')
def func(item):
    raw_dir_path, file_name , label = item
    raw_file_path = Path(raw_dir_path) / file_name
    img = cv2.imread(raw_file_path, cv2.IMREAD_GRAYSCALE)
    row_min, col_min = round(img.shape[0]*0.05), round(img.shape[1]*0.05)
    row_max, col_max = img.shape[0]-row_min, img.shape[1]-col_min
    img = img[row_min:row_max, col_min:col_max]
    _, img = cv2.threshold(img, 235, 255, cv2.THRESH_BINARY_INV)
    col_min, row_min, height, width = cv2.boundingRect(img)
    col_max, row_max = col_min+height, row_min+width
    img = img[row_min:row_max, col_min:col_max]
    if len(img) > 0:
        img = cv2.resize(img, dsize=(54,54))
        img = cv2.copyMakeBorder(
            img, 5, 5, 5, 5,
            cv2.BORDER_CONSTANT,
            value=[0]
        )
        img = cv2.cvtColor(img, cv2.IMREAD_COLOR)
        save_path = SAVE_PATH / file_name
        cv2.imwrite(save_path, img)
        encoder = joblib.load('data/labelEncoder.bin')
        label = encoder.transform([label]).tolist()
        return [str(save_path), label]
pool = Pool(8)
data_info = pool.map(func, data_info)
pool.close()
pool.join()
print('Create Train Images Done.')

data_info = [item for item in data_info if item is not None]
with open('data/dataInfo.json', 'w') as f:
    json.dump(data_info, f, ensure_ascii=False)
