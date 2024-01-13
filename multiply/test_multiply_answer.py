#!/usr/bin/python3
# -*- coding: utf8 -*-

import pandas as pd
import numpy as np

# Generate data
train = []
for x1 in range(2, 10):
    for x2 in range(1, 10):
        y = x1*x2
        train.append([x1, x2, y])
train = pd.DataFrame(train, columns=['x1', 'x2', 'y'])

test = []
for x1 in range(102, 110):
    for x2 in range(101, 110):
        y = x1*x2
        test.append([x1, x2, y])
test = pd.DataFrame(test, columns=['x1', 'x2', 'y'])

# Create dataframe for result
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
result = pd.DataFrame([])

# Linear Regression
from sklearn.linear_model import LinearRegression

train = np.log(train)
test = np.log(test)

linear_model = LinearRegression()
linear_model.fit(train[['x1','x2']], train['y'])
train['predict_linear'] = linear_model.predict(train[['x1','x2']])
test['predict_linear'] = linear_model.predict(test[['x1','x2']])

train = np.exp(train)
test = np.exp(test)

result.loc['Linear', 'R-Squared(train)'] = r2_score(train['y'], train['predict_linear'])
result.loc['Linear', 'RMSE(train)'] = \
    mean_squared_error(train['y'], train['predict_linear'], squared=False)
result.loc['Linear', 'R-Squared(test)'] = r2_score(test['y'], test['predict_linear'])
result.loc['Linear', 'RMSE(test)'] = \
    mean_squared_error(test['y'], test['predict_linear'], squared=False)

# GBM
from xgboost import XGBRegressor

xgb_model = XGBRegressor(n_jobs=1)
xgb_model.fit(train[['x1','x2']], train['y'])

train['predict_gbm'] = xgb_model.predict(train[['x1','x2']])
train['predict_gbm'] = train['predict_gbm'].round(1)
test['predict_gbm'] = xgb_model.predict(test[['x1','x2']])
test['predict_gbm'] = test['predict_gbm'].round(1)

result.loc['GBM', 'R-Squared(train)'] = r2_score(train['y'], train['predict_gbm'])
result.loc['GBM', 'RMSE(train)'] = \
    mean_squared_error(train['y'], train['predict_gbm'], squared=False)
result.loc['GBM', 'R-Squared(test)'] = r2_score(test['y'], test['predict_gbm'])
result.loc['GBM', 'RMSE(test)'] = \
    mean_squared_error(test['y'], test['predict_gbm'], squared=False)

# Neural Network - Pytorch
import torch
from torch import nn, FloatTensor
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
import lightning as L
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import RichProgressBar

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
        
    def forward(self, x):
        return self.func(x)

class Model(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            Lambda(lambda x: torch.log(x)),
            nn.Linear(2, 1),
            Lambda(lambda x: torch.exp(x))
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss_function = nn.MSELoss()
        loss = loss_function(self.forward(x), y)
        self.log('loss', loss)
        return loss
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.1)

dataset = TensorDataset(
    FloatTensor(train[['x1','x2']].values),
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

train['predict_nn'] = pytorch_model(
    FloatTensor(train[['x1','x2']].values)
).detach().numpy()
train['predict_nn'] = train['predict_nn'].round(1)
test['predict_nn'] = pytorch_model(
    FloatTensor(test[['x1','x2']].values)
).detach().numpy()
test['predict_nn'] = test['predict_nn'].round(1)

result.loc['NN', 'R-Squared(train)'] = r2_score(train['y'], train['predict_nn'])
result.loc['NN', 'RMSE(train)'] = \
    mean_squared_error(train['y'], train['predict_nn'], squared=False)
result.loc['NN', 'R-Squared(test)'] = r2_score(test['y'], test['predict_nn'])
result.loc['NN', 'RMSE(test)'] = \
    mean_squared_error(test['y'], test['predict_nn'], squared=False)

print(train)
print(test)
print(result.round(2))