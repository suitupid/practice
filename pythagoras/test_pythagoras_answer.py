#!/usr/bin/env python3

import pandas as pd
import torch
from torch import nn, FloatTensor
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
import lightning as L
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import RichProgressBar
from sklearn.metrics import mean_squared_error

train = []
for x1 in range(2, 10):
    for x2 in range(1, 10):
        y = (x1**2)+(x2**2)
        y = y**(1/2)
        train.append([x1, x2, y])
train = pd.DataFrame(train, columns=['x1', 'x2', 'y'])
test = []
for x1 in range(102, 110):
    for x2 in range(101, 110):
        y = (x1**2)+(x2**2)
        y = y**(1/2)
        test.append([x1, x2, y])
test = pd.DataFrame(test, columns=['x1', 'x2', 'y'])

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
        
    def forward(self, x):
        return self.func(x)

class Model(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc_x1 = nn.Sequential(
            nn.Linear(1, 2, bias=False),
            Lambda(lambda x: torch.log(x)),
            nn.Linear(2, 1, bias=False)
        )
        self.fc_x1.apply(self.initialize_weights)
        self.fc_x2 = nn.Sequential(
            nn.Linear(1, 2, bias=False),
            Lambda(lambda x: torch.log(x)),
            nn.Linear(2, 1, bias=False)
        )
        self.fc_x2.apply(self.initialize_weights)
        self.fc = nn.Sequential(
            Lambda(lambda x: torch.exp(x)),
            nn.Linear(2, 1, bias=False),
            Lambda(lambda x: x**(1/2))
        )
        self.fc.apply(self.initialize_weights)

    def initialize_weights(self, layer):
        if isinstance(layer, torch.nn.Linear):
            nn.init.uniform_(layer.weight, 0, 2)
    
    def forward(self, x):
        x1 = x[:, 0].reshape(-1, 1)
        x2 = x[:, -1].reshape(-1, 1)
        x1 = self.fc_x1(x1)
        x2 = self.fc_x2(x2)
        out = torch.concatenate([x1, x2], axis=1)
        out = self.fc(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss_function = nn.MSELoss()
        loss = loss_function(self.forward(x), y)
        self.log('loss', loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-2)

dataset = TensorDataset(
    FloatTensor(train[['x1', 'x2']].values),
    FloatTensor(train[['y']].values)
)
dataloader = DataLoader(dataset, batch_size=len(dataset))
pytorch_model = Model()
trainer = Trainer(
    max_epochs=10000, accelerator='cpu',
    callbacks=[
        EarlyStopping(monitor='loss', patience=100),
        RichProgressBar()
    ]
)
trainer.fit(pytorch_model, dataloader)

train['predict'] = pytorch_model(FloatTensor(train[['x1', 'x2']].values)).detach().numpy()
print(mean_squared_error(train['y'], train['predict'], squared=False))

test['predict'] = pytorch_model(FloatTensor(test[['x1', 'x2']].values)).detach().numpy()
print(mean_squared_error(test['y'], test['predict'], squared=False))