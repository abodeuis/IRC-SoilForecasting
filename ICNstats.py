import os
import sys
import logging
import configparser

import json
import numpy as np
import pandas as pd
#import statsmodels.api as sm
import matplotlib.pyplot as plt

#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.python.keras import layers

from tqdm import tqdm
from datetime import datetime
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

log = logging.getLogger('ICNstats')

numeric_cols = ['year', 'month', 'day', 'max_wind_gust', 'avg_wind_speed', 'avg_wind_dir', 'sol_rad', 'max_air_temp', 'min_air_temp', 'avg_air_temp', 'max_rel_hum', 'min_rel_hum', 'avg_rel_hum', 'avg_dewpt_temp', 'precip', 'pot_evapot', 'max_soiltemp_4in_sod', 'min_soiltemp_4in_sod', 'avg_soiltemp_4in_sod', 'max_soiltemp_8in_sod', 'min_soiltemp_8in_sod', 'avg_soiltemp_8in_sod', 'max_soiltemp_4in_bare', 'min_soiltemp_4in_bare', 'avg_soiltemp_4in_bare', 'max_soiltemp_2in_bare', 'min_soiltemp_2in_bare', 'avg_soiltemp_2in_bare', 'SM5', 'SM10', 'SM20', 'SM50', 'SM100', 'SM150']

class Config:
    class Default:
        # [Data]
        data_source = ''
        prediction_target = 'avg_soiltemp_4in_sod' # Name of the column that we are going to try to predict
        validation_percent = 0.2 # Percentage of training set to use as validation

        # [Training]
        epochs = 5          # Number of training epochs
        batch_size = 32     # Number of individual samples to use for each training step.
        early_stopping = 3  # Early stopping, will stop training if no improvement in 'N' epochs
        validation_batchs = 3 # Number of batchs to use per training step as validation
        learning_rate = 1e-3 # learning rate

        # [Plots]
        sample_year = 2018  # Year to sample for plots
        acf_days = 14       # Amount of days to forecast out for the ACF PACF plots

        # [Other]
        debug_level = 'INFO'
        save_path =  os.path.join('models', datetime.now().strftime('model_%d%m%Y_%H%M%S.pkl'))# Path to save the model to

        @staticmethod
        def write(file):
            content = ("# This is the configuration file for ICNstats.py.\n" +
                       "\n" +
                       "[Data]\n" +
                       "# The directory or files to load the data from\n" +
                       "data_source = ['example.csv','other_example.txt','example_dir']\n"
                       "# Name of the column that is going to be predicted\n" +
                       "prediction_target = \'{}\'\n".format(Config.Default.prediction_target) +
                       "# Percentage of the data to use for validation\n" +
                       "validation_percent = {}\n".format(Config.Default.validation_percent) +
                       "\n" +
                       "[Training]\n" +
                       "# Number of epochs to train for.\n" +
                       "epochs = {}\n".format(Config.Default.epochs) +
                       "batch_size = {}\n".format(Config.Default.batch_size) +
                       "early_stopping = {}\n".format(Config.Default.early_stopping) +
                       "validation_batchs = {}\n".format(Config.Default.validation_batchs) +
                       "learning_rate = {}\n".format(Config.Default.learning_rate) + 
                       "\n" +
                       "[Plots]\n" +
                       "# Year to sample for plots\n" +
                       "sample_year = {}\n".format(Config.Default.sample_year) +
                       "# Amount of days to plot out for the ACF plots\n" +
                       "acf_days = {}\n".format(Config.Default.acf_days) +
                       "\n" +
                       "[Other]\n" +
                       "debug_level = {}\n".format(Config.Default.debug_level) +
                       "save_path = \n")
        
            with open(file,'w') as fh:
                fh.write(content)

    def __init__(self, filepath):
        self.filepath = filepath
        self.load(filepath)
        self.validate()

    def load(self, filepath):
        log.info('Loading config from \"{}\"'.format(filepath))
        if not os.path.isfile(filepath):
            log.warning('No config file found at \"{}\". Using defaults instead'.format(filepath))
            example_filepath = os.path.join(os.path.dirname(filepath), "example_config.ini")
            if not os.path.isfile(example_filepath):
                log.warning('Example config not present, creating it at \"{}\"'.format(example_filepath))
                Config.Default.write(example_filepath)
            
        config = configparser.ConfigParser()
        config.read(filepath)

        # [Data]
        self.data_source = config.get('Data','data_source', fallback=Config.Default.data_source)
        self.prediction_target = config.get('Data','prediction_target', fallback=Config.Default.prediction_target)
        self.validation_percent = config.get('Data','validation_percent', fallback=Config.Default.validation_percent)

        # [Training]
        self.epochs = config.get('Training','epochs', fallback=Config.Default.epochs)
        self.batch_size = config.get('Training','batch_size', fallback=Config.Default.batch_size)
        self.early_stopping = config.get('Training','early_stopping', fallback=Config.Default.early_stopping)
        self.validation_batchs = config.get('Training','validation_batchs', fallback=Config.Default.validation_batchs)
        self.learning_rate = config.get('Training','learning_rate', fallback=Config.Default.learning_rate)

        # [Plots]
        self.sample_year = config.get('Plots', 'sample_year', fallback=Config.Default.sample_year)
        self.acf_days = config.get('Plots', 'acf_days', fallback=Config.Default.acf_days)

        # [Other]
        self.debug_level = config.get('Other','debug_level', fallback=Config.Default.debug_level)
        self.save_path = config.get('Other','save_path', fallback=Config.Default.save_path)

    def validate(self):
        # Trim quotes and convert to list if needed
        if self.data_source == '':
            log.critical('No data source was given. Cannot run without any data')
            exit(1)
        if '[' in self.data_source:
            self.data_source = json.loads(self.data_source.replace('\'', '"'))
            self.data_source = [f.replace('"','') for f in self.data_source]
        else:
            self.data_source = self.data_source.replace('\'', '').replace('"', '')
        self.prediction_target = self.prediction_target.replace('\'', '').replace('"', '')
        self.save_path = self.save_path.replace('\'', '').replace('"', '')

        # Convert strings to numeric values
        self.validation_percent = float(self.validation_percent)

        self.epochs = int(self.epochs)
        self.batch_size = int(self.batch_size)
        self.early_stopping = int(self.early_stopping)
        self.validation_batchs = int(self.validation_batchs)
        self.learning_rate = float(self.learning_rate)

        self.sample_year = int(self.sample_year)
        self.acf_days = int(self.acf_days)

        # Convert debugLevel string to logging enum
        if hasattr(logging, self.debug_level.upper()):
            self.debug_level = getattr(logging, self.debug_level.upper())
        else:
            log.warning('Invalid debug_level \"{}\" given, falling back to {}'.format(self.debug_level, Config.Default.debug_level))
            self.debug_level = Config.Default.debug_level

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

def ingest_txt_file(file, keep_estimated=False):
    logging.info("Ingesting Data Files")
    error_cols = ['xwser', 'awser', 'awder', 'soler', 'xater', 'nater', 'aater', 'xrher', 'nrher', 'arher', 'adper', 'pcer', 'pevaper', 'xst4soder', 'nst4soder', 'ast4soder', 'xst8soder', 'nst8soder', 'ast8soder', 'xst4bareer', 'nst4bareer', 'ast4bareer', 'xst2bareer', 'nst2bareer', 'ast2bareer']

    # Read data files
    data = pd.DataFrame()
    
    log.debug('Reading {}'.format(file))
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

def load_data(filepath, val_percent):
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
            file_data, log_df, log_dn, log_dm, log_de = ingest_txt_file(file)
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
    data = data[numeric_cols].apply(pd.to_numeric, downcast='float', errors='coerce')

    # Split into Train and Validation sets
    thresh = round(len(data)*val_percent)
    train = data[:thresh]
    val = data[thresh:]

    log.info('Finished loading data\n\tTrain data {} samples\n\tVal data {} samples'.format(len(train), len(val)))  
    log.info('Dropped {} missing values, {} estimated values, {} null values, and {} footer rows'.format(dropped_missing, dropped_estimated, dropped_null, dropped_footer))

    return train, val

def plot_corralation(df, savepath='Correlation.png'):
    f = plt.figure(figsize=(19, 19))
    plt.matshow(df.corr(), fignum=f.number)
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=45)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)
    plt.savefig(savepath)

def data_analysis(data, config):
    # Stats analaysis
    log.info('Starting data analysis')
    year_sample = data[data['year'] == config.sample_year]
    year_sample.reset_index()

    log.info('Creating stats csv')
    # Write stats to a csv
    statsdf = data.describe()
    statsdf.to_csv(os.path.join(config.save_path, 'stats.csv'))

    log.info('Plotting Corralation graph')
    # Plot Corralation graph
    if not os.path.exists(os.path.join(config.save_path, 'plots')):
        os.makedirs(os.path.join(config.save_path, 'plots'))
    plot_corralation(data, os.path.join(config.save_path, 'plots', 'Correlation.png'))

    log.info('Creating ACF, PACF plots for each variable')
    # Plot ACF, PACF for each column
    
    days=config.acf_days
    for key in tqdm(numeric_cols):
        if key in ['year','month','day']:
            continue
        # Orginal Series
        fig, axes = plt.subplots(3,1, figsize=(20,14), sharex=True)
        axes[0].plot(year_sample[key]); axes[0].set_title('Original Series, Year {}'.format(config.sample_year))
        #plot_acf(year_sample[key], ax=axes[0,1], lags=days)

        # 1st Differencing
        axes[1].plot(year_sample[key].diff().dropna()); axes[1].set_title('1st Differential')
        #plot_acf(year_sample[key].diff().dropna(), ax=axes[1,1], lags=days)

        # 2nd Differencing
        axes[2].plot(year_sample[key].diff().diff().dropna()); axes[2].set_title('2nd Differential')
        #plot_acf(year_sample[key].diff().diff().dropna(), ax=axes[2,1], lags=days)

        plt.savefig(os.path.join(config.save_path, 'plots', 'Diff_{}.png'.format(key)))
        plt.close()
        try:
            acf_fig = plot_acf(year_sample[key], lags=days, title=('ACF'))
            pacf_fig = plot_pacf(year_sample[key], method='ols', lags=days, title=('PACF'))
            acf_fig.savefig(os.path.join(config.save_path, 'plots', 'ACF_{}.png'.format(key)))
            pacf_fig.savefig(os.path.join(config.save_path, 'plots', 'PACF_{}.png'.format(key)))
        except:
            log.warning('Could not generate ACF graph for {}'.format(key))
            #log.exception("An error occured during variable {}".format(key))
            continue
    
    log.info('Finished data analysis')

def main():
    # Start logger
    setup_logger(os.path.join('logs', datetime.now().strftime('%d%m%Y_%H%M%S.log')), logging.DEBUG)
    
    # Load User Config
    config = Config("config.ini")
    set_log_level(config.debug_level)
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)

    # Load Dataset
    train, val = load_data(config.data_source, (1-config.validation_percent))

    # Pre model analysis
    data_analysis(pd.concat([train,val]), config)
    

if __name__ == '__main__':
    main()