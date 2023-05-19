import os
import logging
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# Should the torch bit be in here?
import torch

log = logging.getLogger('ICN-Forecasting')

def ingest_txt_file(file, error_cols=[], keep_estimated=False):
    logging.info("Ingesting Data Files")
    
    # Read data files
    data = pd.DataFrame()
    
    df = pd.read_csv(file, sep='\t')

    # Remove ICN footer
    findex = 0
    for row in df.year.iloc[-20:]:
        if not pd.isna(row) and'M = Missing Data' in row:
            df.drop(df.tail(20-findex).index, inplace=True)
            log.debug('Dropped {} ICN footer rows'.format(20-findex))
            dropped_footer = 20-findex
            break
        findex += 1
        
    # Drop Null values
    count = len(df)
    df.dropna(subset=['year','month','day'], inplace=True)
    log.debug('Dropped {} null values'.format(count-len(df)))
    dropped_null = count-len(df)

    # Remove Missing values
    count = len(df)
    for col in error_cols:
        df = df[df[col] != 'M']
    log.debug('Dropped {} missing values'.format(count-len(df)))
    dropped_missing = count-len(df)

    # Remove Estimated values
    count = len(df)
    if not keep_estimated:
        for col in error_cols:
            df = df[df[col] != 'E']
        log.debug('Dropped {} estimated values'.format(count-len(df)))
        dropped_estimated = count-len(df)

    # Append file
    data = pd.concat([data, df])

    return data, dropped_footer, dropped_null, dropped_missing, dropped_estimated

def load_data(filepath, numeric_cols, error_cols=[]):
    # If its a single string change it to a list
    if isinstance(filepath, str):
        filepath = [filepath]

    # Expand directories
    tmp = []
    for file in filepath:
        if os.path.isdir(file):
            for f in os.listdir(file):
                name, ext = os.path.splitext(f)
                if ext in ['.txt', '.csv']:
                    tmp.append(os.path.join(file,f))
        else:
            tmp.append(file)
    filepath = tmp

    log.info('Loading {} data files'.format(len(filepath)))
    data = pd.DataFrame()
    dropped_footer = 0
    dropped_missing = 0
    dropped_estimated = 0
    dropped_null = 0
    # Load each file
    for file in tqdm(filepath):
        name, ext = os.path.splitext(file)
        if ext == '.txt':
            file_data, log_df, log_dn, log_dm, log_de = ingest_txt_file(file, error_cols=error_cols)
            data = pd.concat([data, file_data])
            dropped_footer += log_df
            dropped_null += log_dn
            dropped_missing += log_dm
            dropped_estimated += log_de
        elif ext == '.csv':
            file_data = pd.read_csv(file)
            data = pd.concat([data, file_data])
        else:
            log.warning('Unable to load {}, unknown extension "{}". Only .txt and .csv are supported'.format(file, ext))

    # Convert string values to numeric
    valid_cols = [col for col in numeric_cols if col in data]
    invalid_cols = [col for col in numeric_cols if col not in data]
    if len(invalid_cols) > 0:
        log.warning('Numeric columns : {} were specified but not present in the data'.format(invalid_cols))
    converted_data = data[valid_cols].apply(pd.to_numeric, downcast='float', errors='coerce')

    # Check that there are any numeric columns
    if len(valid_cols) < 1:
        log.critical('No numeric columns were present in the data. No data to work with exiting now.')
        exit(1)
    
    # Convert hourly null value (9999) to python null
    if 'HDATE' in data:
        converted_data[converted_data > 9001] = pd.NA

    # Convert datetime format to timestamp column
    if 'HDATE' in data: # Hourly data format
        converted_data['timestamp'] = data.apply(lambda x: datetime.strptime(x['HDATE'],'%m/%d/%Y %H:%M'), axis=1)
    elif 'year' in data and 'month' in data and 'day' in data: # Daily data format
        converted_data['timestamp'] = data.apply(lambda x: datetime(int(x['year']),int(x['month']),int(x['day'])), axis=1)
    else:
        log.warning('No datetime format could be found in the data. (No HDATE or year month day combo)')

    # Keep site tag
    if 'station' in data: # Hourly data format
        data['site'] = data['station']
    converted_data['site'] = data['site']

    log.info('Finished loading data with {} samples present'.format(len(data)))  
    log.info('Dropped {} missing values, {} estimated values, {} null values, and {} footer rows'.format(dropped_missing, dropped_estimated, dropped_null, dropped_footer))

    return converted_data

def create_lstm_dataset(data, target_col, window_size):
    x, y = [], []
    for i in range(len(data)-window_size):
        feature = data[i:i+window_size]
        target = data[target_col][i+1:i+window_size+1]
        # Skip windows with any null values.
        if feature.isnull().values.any() or target.isnull().values.any():
            continue
        x.append(feature.to_numpy())
        y.append(target.to_numpy()) 
    return torch.tensor(x), torch.tensor(y)

def create_dataset(data, target_col, training_size, prediction_size):
    x, y = [], []
    for i in range(len(data)-(training_size+prediction_size)):
        feature = data[i:i+training_size]
        target = data[target_col][i+training_size-prediction_size:i+training_size+prediction_size]
        # Skip windows with any null values.
        if feature.isnull().values.any() or target.isnull().values.any():
            continue
        x.append(feature.to_numpy())
        y.append(target.to_numpy()) 
    return torch.tensor(x), torch.tensor(y)