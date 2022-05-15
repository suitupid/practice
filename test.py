#!/usr/bin/python3
# -*- coding: utf8 -*-

import numpy as np
import pandas as pd
from random import randint, random
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Generate data
random_int_x1 = randint(1, 5)
random_int_x2 = randint(1, 5)
random_int_x3 = randint(1, 5)

train = []
for x1 in range(1, 10):
    for x2 in range(1, 10):
        for x3 in range(1, 10):
            x4 = randint(0, 1)
            if x4 == 1:
                x4 = 'Y'
                y = (random_int_x1*x1)+(random_int_x2*x2)+(random_int_x3*x3)
            else:
                x4 = 'N'
                y = (random_int_x1*x1)*(random_int_x2*x2)*(random_int_x3*x3)
            train.append([
                np.round(x1+(randint(-1, 1)*random()/2.1), 2),
                np.round(x2+(randint(-1, 1)*random()/2.1), 2),
                np.round(x3+(randint(-1, 1)*random()/2.1), 2),
                x4,
                y
            ])
train = pd.DataFrame(train, columns=['x1', 'x2', 'x3', 'x4', 'y'])

test = []
for x1 in range(101, 110):
    for x2 in range(101, 110):
        for x3 in range(101, 110):
            x4 = randint(0, 1)
            if x4 == 1:
                x4 = 'Y'
                y = (random_int_x1*x1)+(random_int_x2*x2)+(random_int_x3*x3)
            else:
                x4 = 'N'
                y = (random_int_x1*x1)*(random_int_x2*x2)*(random_int_x3*x3)
            test.append([
                np.round(x1+(randint(-1, 1)*random()/2.1), 2),
                np.round(x2+(randint(-1, 1)*random()/2.1), 2),
                np.round(x3+(randint(-1, 1)*random()/2.1), 2),
                x4,
                y
            ])
test = pd.DataFrame(test, columns=['x1', 'x2', 'x3', 'x4', 'y'])

# Pre-processing
train['x4'] = train['x4'].replace({'Y':1, 'N':0})
test['x4'] = test['x4'].replace({'Y':1, 'N':0})

# Create dataframe for result
result = pd.DataFrame([], columns=['R-Squared', 'MSE'])

# Linear Regression
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()
linear_model.fit(train[['x1','x2','x3','x4']], train['y'])
train['predict_linear'] = linear_model.predict(train[['x1','x2','x3','x4']])

result.loc['Linear', 'R-Squared'] = r2_score(train['y'], train['predict_linear'])
result.loc['Linear', 'MSE'] = \
    mean_squared_error(train['y'], train['predict_linear'], squared=False)

# Neural Network - tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.python.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.05
K.set_session(tf.compat.v1.Session(config=config))

tensorflow_model = Sequential()
tensorflow_model.add( Dense(128, input_dim=4, activation='relu') )
tensorflow_model.add( Dense(1, activation='linear') )
tensorflow_model.compile(
    loss='mse',
    optimizer=Adam(learning_rate=0.1),
    metrics=['mse']
)
tensorflow_model.fit(
    train[['x1','x2','x3','x4']], train['y'],
    batch_size=len(train),
    epochs=1000,
    callbacks=[EarlyStopping(patience=10, monitor='loss')],
    verbose=0
)
train['predict_tensorflow'] = tensorflow_model.predict(train[['x1','x2','x3','x4']])

result.loc['Tensorflow', 'R-Squared'] = r2_score(train['y'], train['predict_tensorflow'])
result.loc['Tensorflow', 'MSE'] = \
    mean_squared_error(train['y'], train['predict_tensorflow'], squared=False)

# Neural Network - pytorch
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

class model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.mse_loss(self.forward(x), y, reduction='mean')
        self.log('loss', loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.1)
data_tensor = TensorDataset(
    torch.from_numpy(train[['x1','x2','x3','x4']].values).float(),
    torch.from_numpy(train[['y']].values).float(),
)
train_dataloader = DataLoader(data_tensor, batch_size=len(train))
pytorch_model = model()
trainer = Trainer(
    max_epochs=1000, gpus=-1,
    enable_progress_bar=False, logger=False, enable_checkpointing=False,
    callbacks=[EarlyStopping(monitor='loss', patience=10)]
)
trainer.fit(pytorch_model, train_dataloader)
train['predict_pytorch'] = pytorch_model(
    torch.from_numpy(train[['x1','x2','x3','x4']].values).float()
).detach().numpy()

result.loc['Pytorch', 'R-Squared'] = r2_score(train['y'], train['predict_pytorch'])
result.loc['Pytorch', 'MSE'] = \
    mean_squared_error(train['y'], train['predict_pytorch'], squared=False)

# XGB
from xgboost import XGBRegressor

xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.1, n_jobs=8)
xgb_model.fit(train[['x1','x2','x3','x4']], train['y'])
train['predict_xgboost'] = xgb_model.predict(train[['x1','x2','x3','x4']])

result.loc['XGB', 'R-Squared'] = r2_score(train['y'], train['predict_xgboost'])
result.loc['XGB', 'MSE'] = \
    mean_squared_error(train['y'], train['predict_xgboost'], squared=False)

# print result
result