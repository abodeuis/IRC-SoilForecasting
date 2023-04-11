import os
import sys
import logging
import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_data

from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
from tqdm import tqdm

# Internal Project files
import src.analysis as analysis
from src.config import Config
import src.dataloader as icn_data
from src.models import LSTM_Model

log = logging.getLogger('ICN-Forecasting')

def parse_command_line():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-c', '--config', default='config.ini', action='store', help='Config file to load from')

    args = parser.parse_args()
    return args

def setup_logger(filepath, debuglvl):
    # Create directory if necessary
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))

    # Create Formatter
    log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(filename)s:(%(lineno)d) - %(message)s', datefmt='%d/%m/%Y %H:%M:%S')

    # Setup File handler
    file_handler = logging.FileHandler(filepath)
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(debuglvl)

    # Setup Stream handler (i.e. console)
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(log_formatter)
    stream_handler.setLevel(debuglvl)

    # Add Handlers to logger
    log.addHandler(file_handler)
    log.addHandler(stream_handler)
    log.setLevel(debuglvl)

def set_log_level(loglvl):
    for h in log.handlers:
        h.setLevel(loglvl)
    log.setLevel(loglvl)

def train(model, opt, loss_fn, train_dataloader, val_dataloader, config, tb):
    best_model = ''
    best_loss = np.Infinity
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
                val_err = np.sqrt(loss_fn(ŷ, y))
            #log.info('Epoch {}: Val error = {}'.format(epoch, val_err))
            tb.add_scalar('Loss', loss, epoch)
            #tb.add_scaler('Target Accuracy', np.sqrt(loss_fn()), epoch)
            pbar.set_description('Epoch {}: Train Error = {tr:.2f}, Val Error = {vr:.2f}'.format(epoch, tr=train_err, vr=val_err))
            #print('Epoch {}: Val error = {}'.format(epoch, val_err))
        # Early Stopping
        if train_err < best_loss:
            best_loss = train_err
            best_model = model
            es = 0
        if es > config.early_stopping:
            log.info('Stopping early due to no improvement in loss in {} epochs'.format(config.early_stopping))
            break
        es = es + 1
    
    return best_model

def main():
    args = parse_command_line()

    # Start logger
    setup_logger(os.path.join('logs', datetime.now().strftime('%d%m%Y_%H%M%S.log')), logging.DEBUG)
    tb = SummaryWriter()

    # Load User Config
    config = Config(args.config)
    set_log_level(config.debug_level)
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)

    # Load Dataset
    data = icn_data.load_data(config.data_source, config.numeric_cols, config.error_cols)

    # Pre model analysis
    analysis.data_analysis(data, config)

    # Split into Train and Validation sets
    # TODO better random sampling of the test and validation set.
    thresh = round(len(data)*(1-config.validation_percent))
    train_data = data[:thresh]
    val_data = data[thresh:]

    train_x, train_y = icn_data.create_dataset(train_data[config.numeric_cols], config.prediction_target, 28)
    val_x, val_y = icn_data.create_dataset(val_data[config.numeric_cols], config.prediction_target, 28)

    # Naive Model (Guess last years answer)
    
    # linear regression

    # RNN

    # LSTM
    log.debug('Creating model with data_shape of {}'.format(len(train_data[config.numeric_cols].keys())))
    model = LSTM_Model(len(train_data[config.numeric_cols].keys()))
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    loss = nn.MSELoss()
    train_loader = torch_data.DataLoader(torch_data.TensorDataset(train_x, train_y), shuffle=True, batch_size=config.batch_size)
    val_loader = torch_data.DataLoader(torch_data.TensorDataset(val_x, val_y), shuffle=True, batch_size=config.validation_batchs)

    train(model, optimizer, loss, train_loader, val_loader, config, tb)

    # GRU

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

if __name__ == '__main__':
    main()