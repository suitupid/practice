#!/usr/bin/python3
# -*- coding: utf8 -*-

import json
import math

import torch
from torch import nn
import lightning as L


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)


class CustomNetwork(L.LightningModule):
    def __init__(self):
        super().__init__()

        with open('data/stoi.json', 'r', encoding='utf-8') as f:
            self.stoi = json.load(f)
        self.vocab_size = len(self.stoi)
        self.tag_size = 2
        self.d_model = 256
        self.nhead = 8
        self.num_layers = 4
        self.lr = 1e-4

        self.embedding = nn.Embedding(
            self.vocab_size,
            self.d_model,
            padding_idx=self.stoi['<PAD>']
        )
        self.pos_encoder = PositionalEncoding(self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )

        self.fc = nn.Linear(self.d_model, self.tag_size)

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=-100,
            weight=torch.tensor([1.0, 5.0])
        )

    def forward(self, x):
        padding_mask = (x == 0)
        emb = self.embedding(x)
        emb = self.pos_encoder(emb)
        out = self.transformer(emb, src_key_padding_mask=padding_mask)
        out = self.fc(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(
            logits.reshape(-1, logits.size(-1)),
            y.view(-1)
        )
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(
            logits.reshape(-1, logits.size(-1)),
            y.view(-1)
        )
        self.log('val_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
