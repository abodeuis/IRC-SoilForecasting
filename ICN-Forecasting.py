import os
import sys
import logging
import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_data

from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
from tqdm import tqdm

# Internal Project files
import src.logging_utils as log_utils
import src.analysis as analysis
from src.config import Config
import src.dataloader as icn_data
from src.models import LSTM_Model, RNN_Model, GRU_Model, Dense_Model

log = logging.getLogger('ICN-Forecasting')

def parse_command_line():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-c', '--config', default='config.ini', action='store', help='Config file to load from')
    args = parser.parse_args()
    return args

def train(model, opt, loss_fn, train_dataloader, val_dataloader, config, tb):
    log.info('Starting training of {}.'.format(model.model_name))
    best_model = ''
    best_train_loss = np.Infinity
    best_val_loss = np.Infinity
    es = 0
    val_err = 0
    pbar = tqdm(range(config.epochs))
    pbar.set_description('Epoch  : Train Error =   , Val Error = ')
    sample, _ = next(iter(train_dataloader))
    tb.add_graph(model, sample)
    for epoch in pbar:
        model.train()
        for x, y in train_dataloader:
            ŷ = model(x).squeeze()
            #print('{},{}'.format(ŷ.squeeze(), y))
            loss = loss_fn(ŷ, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        # Validation
        model.eval()
        with torch.no_grad():
            ŷ = model(x).squeeze()
            train_err = np.sqrt(loss_fn(ŷ, y))
            if epoch % 5 == 0:
                x, y = next(iter(val_dataloader))
                ŷ = model(x).squeeze()
                val_loss = loss_fn(ŷ, y)
                val_err = np.sqrt(val_loss)
                tb.add_scalar('{} : Val Loss'.format(model.model_name), val_loss, epoch)
            #log.info('Epoch {}: Val error = {}'.format(epoch, val_err))
            tb.add_scalar('{} : Train Loss'.format(model.model_name), loss, epoch)
            #tb.add_histogram('lstm.bias', model.lstm.bias, epoch)
            #tb.add_histogram('lstm.weight', model.lstm.weight, epoch)
            #tb.add_scaler('Target Accuracy', np.sqrt(loss_fn()), epoch)
            pbar.set_description('Epoch {}: Train Error = {tr:.2f}, Val Error = {vr:.2f}'.format(epoch, tr=train_err, vr=val_err))
            #print('Epoch {}: Val error = {}'.format(epoch, val_err))
        # Early Stopping
        if train_err < best_train_loss:
            best_train_loss = train_err
            best_val_loss = val_err
            best_model = model
            es = 0
        if es > config.early_stopping:
            log.warning('Stopping {} training early due to no improvement in train loss in {} epochs'.format(model.model_name, config.early_stopping))
            break
        es = es + 1
    log.info('Training ended after {} epochs with train loss {tl:.2f}, val loss {vl:.2f}'.format(epoch, tl=best_train_loss, vl=best_val_loss))
    return best_model

#def main():
args = parse_command_line()

# Start logger
log_utils.setup_logger('Latest.log', logging.DEBUG)

# Load User Config
config = Config(args.config)

if config.save_path != '' and not os.path.exists(os.path.join(config.save_path, 'logs')):
    os.makedirs(os.path.join(config.save_path, 'logs'))
log_utils.set_log_file(os.path.join(config.save_path, 'logs', 'training.log'))
log_utils.set_log_level(config.debug_level)
# TensorBoard Logger
tb = SummaryWriter(os.path.join(config.save_path, 'logs'))

# Load Dataset
data = icn_data.load_data(config.data_source, config.numeric_cols, config.error_cols)

# Pre model analysis
#analysis.data_analysis(data, config)

# Split into Train and Validation sets
# TODO better random sampling of the test and validation set.
thresh = round(len(data)*(1-config.validation_percent))
train_data = data[:thresh]
val_data = data[thresh:]

train_x, train_y = icn_data.create_dataset(train_data[config.numeric_cols], config.prediction_target, config.training_frame_size, config.prediction_frame_size)
val_x, val_y = icn_data.create_dataset(val_data[config.numeric_cols], config.prediction_target, config.training_frame_size, config.prediction_frame_size)

# Naive Model (Guess last years answer)

# linear regression

# Data Prep for NN
train_loader = torch_data.DataLoader(torch_data.TensorDataset(train_x, train_y), shuffle=True, batch_size=config.batch_size)
val_loader = torch_data.DataLoader(torch_data.TensorDataset(val_x, val_y), shuffle=True, batch_size=config.validation_batchs)
dense_train_loader = torch_data.DataLoader(torch_data.TensorDataset(torch.flatten(train_x, start_dim=1), train_y), shuffle=True, batch_size=config.batch_size)
dense_val_loader = torch_data.DataLoader(torch_data.TensorDataset(torch.flatten(val_x, start_dim=1), val_y), shuffle=True, batch_size=config.batch_size)
input_shape=len(train_data[config.numeric_cols].keys())
model_params = {
    'input_dim' : len(train_data[config.numeric_cols].keys()),
    'hidden_dim' : 50,
    'output_dim' : 1,
    'layers' : 1
}
dense_model_params = {
    'input_dim' : train_x.shape[1] * train_x.shape[2],
    'hidden_dim' : 256,
    'output_dim' : train_y.shape[1],
    'hidden_layers' : 1
}

# Fully Connected Network
log.info('Running Dense Model')
dense_model = Dense_Model(**dense_model_params)
dense_opt = optim.Adam(dense_model.parameters(), lr=config.learning_rate)
loss = nn.MSELoss()
dense_model = train(dense_model, dense_opt, loss, dense_train_loader, dense_val_loader, config, tb)

# RNN
# Input shape = (seq_len, batch, input_size)
log.info('Running RNN Model')
rnn_model = RNN_Model(**model_params)
rnn_opt = optim.Adam(rnn_model.parameters(), lr=config.learning_rate)
loss = nn.MSELoss()
rnn_model = train(rnn_model, rnn_opt, loss, train_loader, val_loader, config, tb)

# LSTM
log.info('Running LSTM Model')
lstm_model = LSTM_Model(**model_params)
lstm_opt = optim.Adam(lstm_model.parameters(), lr=config.learning_rate)
loss = nn.MSELoss()
lstm_model = train(lstm_model, lstm_opt, loss, train_loader, val_loader, config, tb)

# GRU
log.info('Running GRU Model')
gru_model = GRU_Model(**model_params)
gru_opt = optim.Adam(gru_model.parameters(), lr=config.learning_rate)
loss = nn.MSELoss()
gru_model = train(gru_model, gru_opt, loss, train_loader, val_loader, config, tb)

# Predictions
log.info('Generating prediction comparsion plots')
predict = {}
predict_x, predict_y = next(iter(val_loader))
predict['Orginal Series'] = predict_y[0]
with torch.no_grad():
    predict['Dense'] = dense_model(torch.flatten(predict_x[0])).squeeze()
    predict['RNN'] = rnn_model(predict_x[0]).squeeze()
    predict['LSTM'] = lstm_model(predict_x[0]).squeeze()
    predict['GRU'] = gru_model(predict_x[0]).squeeze()

fig, ax = analysis.plot_predictions(predict)
fig.savefig(os.path.join(config.save_path, 'predict.png'))
fig, ax = analysis.plot_error(predict)
fig.savefig(os.path.join(config.save_path, 'error.png'))

# Saving Models
log.info('Saving Models')
if not os.path.exists(os.path.join(config.save_path, 'models')):
    os.makedirs(os.path.join(config.save_path, 'models'))
torch.save(rnn_model, os.path.join(config.save_path, 'models', 'RNN_model.pk'))
torch.save(lstm_model, os.path.join(config.save_path, 'models', 'LSTM_model.pk'))
torch.save(gru_model, os.path.join(config.save_path, 'models', 'GRU_model.pk'))

# ARMA (Auto Regressive Moving Average) model
#arima_model = sm.tsa.ARIMA(data, order=(1,1)).fit()

# ARIMA (Auto Regressive Integrated Moving Average) model
# (p,q,d) p is ar q is ma d is the number of differencing required to make the time series stationary
#arima_model = sm.tsa.ARIMA(train[config.prediction_target], order=(1,1,0)).fit()

# SARIMA

# SARIMAX

# TBATS

#eval_model()

# Augmented Dickey-Fuller test
#sm.tsa.stattools.adfuller()

# Close Tensorboard Writer
tb.close()

#if __name__ == '__main__':
#    main()