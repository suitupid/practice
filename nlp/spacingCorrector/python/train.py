#!/usr/bin/python3
# -*- coding: utf8 -*-

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import RichProgressBar

from dataset import CustomDataset
from model import CustomNetwork


torch.set_float32_matmul_precision('high')

data = []
with open('data/final.jsonl', 'rb') as f:
    offset = 0
    for line in f:
        data.append(offset)
        offset += len(line)

train, valid = train_test_split(data, test_size=0.05, shuffle=True)

train_dataset = CustomDataset(train)
valid_dataset = CustomDataset(valid)
train_dataloader = DataLoader(train_dataset, batch_size=32, num_workers=8)
valid_dataloader = DataLoader(valid_dataset, batch_size=32, num_workers=8)

model = CustomNetwork()
trainer = Trainer(
	max_epochs=10, accelerator='gpu',
	logger=False, enable_checkpointing=False,
	callbacks=[
        EarlyStopping(monitor='val_loss', patience=2),
        RichProgressBar()
    ]
)
trainer.fit(model, train_dataloader, valid_dataloader)

torch.save(model.state_dict(), 'model/spacingCorrector.pt')
