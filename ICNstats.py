import os
import sys
import logging
import configparser

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from dateutil import parser
from datetime import datetime
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

log = logging.getLogger('ICNstats')

class Config:
    class Default:
        # [Data]
        data_source = []
        prediction_target = 'avg_soiltemp_4in_sod' # Name of the column that we are going to try to predict
        validation_percent = 0.2 # Percentage of training set to use as validation
        numeric_cols = ['year', 'month', 'day', 'max_wind_gust', 'avg_wind_speed', 'avg_wind_dir', 'sol_rad', 'max_air_temp', 'min_air_temp', 'avg_air_temp', 'max_rel_hum', 'min_rel_hum', 'avg_rel_hum', 'avg_dewpt_temp', 'precip', 'pot_evapot', 'max_soiltemp_4in_sod', 'min_soiltemp_4in_sod', 'avg_soiltemp_4in_sod', 'max_soiltemp_8in_sod', 'min_soiltemp_8in_sod', 'avg_soiltemp_8in_sod', 'max_soiltemp_4in_bare', 'min_soiltemp_4in_bare', 'avg_soiltemp_4in_bare', 'max_soiltemp_2in_bare', 'min_soiltemp_2in_bare', 'avg_soiltemp_2in_bare', 'SM5', 'SM10', 'SM20', 'SM50', 'SM100', 'SM150']
        error_cols = ['xwser', 'awser', 'awder', 'soler', 'xater', 'nater', 'aater', 'xrher', 'nrher', 'arher', 'adper', 'pcer', 'pevaper', 'xst4soder', 'nst4soder', 'ast4soder', 'xst8soder', 'nst8soder', 'ast8soder', 'xst4bareer', 'nst4bareer', 'ast4bareer', 'xst2bareer', 'nst2bareer', 'ast2bareer']

        # [Training]
        epochs = 5          # Number of training epochs
        batch_size = 32     # Number of individual samples to use for each training step.
        early_stopping = 3  # Early stopping, will stop training if no improvement in 'N' epochs
        validation_batchs = 3 # Number of batchs to use per training step as validation
        learning_rate = 1e-3 # learning rate

        # [Plots]
        sample_period_start = datetime(1900,1,1)
        sample_period_end = datetime(2100,1,1)
        acf_days = 14       # Amount of days to forecast out for the ACF PACF plots

        # [Other]
        debug_level = 'INFO'
        save_path =  os.path.join('models', datetime.now().strftime('model_%d%m%Y_%H%M%S.pkl'))# Path to save the model to

        @staticmethod
        def write(file):
            content = ("# This is the configuration file for simple_rnn.py.\n" +
                       "\n" +
                       "[Data]\n" +
                       "# The directory or files to load the data from\n" +
                       "data_source = ['example.csv','other_example.txt','example_dir']"
                       "# Name of the column that is going to be predicted\n" +
                       "prediction_target = \'{}\'\n".format(Config.Default.prediction_target) +
                       "# Percentage of the data to use for validation\n" +
                       "validation_percent = {}\n".format(Config.Default.validation_percent) +
                       "# The columns that will be used for analysis\n" +
                       "numeric_cols = {}\n".format(Config.Default.numeric_cols) +
                       "# The columns that will be used for flagig error data should contain a 'E' or 'M'\n" +
                       "error_cols = {}\n".format(Config.Default.error_cols) +
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
                       "# Start Time of the sample period. Format is (Month/Day/Year Hour:Min:Sec)\n" +
                       "sample_period_start = \'{}\'\n".format(Config.Default.sample_period_start.strftime('%m/%d/%y %H:%M:%S')) +
                       "# End Time of the sample period\n" +
                       "sample_period_end = \'{}\'\n".format(Config.Default.sample_period_end.strftime('%m/%d/%y %H:%M:%S')) +
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
        self.numeric_cols = config.get('Data','numeric_cols', fallback=Config.Default.numeric_cols)
        self.error_cols = config.get('Data', 'error_cols', fallback=Config.Default.error_cols)

        # [Training]
        self.epochs = config.get('Training','epochs', fallback=Config.Default.epochs)
        self.batch_size = config.get('Training','batch_size', fallback=Config.Default.batch_size)
        self.early_stopping = config.get('Training','early_stopping', fallback=Config.Default.early_stopping)
        self.validation_batchs = config.get('Training','validation_batchs', fallback=Config.Default.validation_batchs)
        self.learning_rate = config.get('Training','learning_rate', fallback=Config.Default.learning_rate)

        # [Plots]
        self.sample_period_start = config.get('Plots', 'sample_period_start', fallback=Config.Default.sample_period_start)
        self.sample_period_end = config.get('Plots', 'sample_period_end', fallback=Config.Default.sample_period_end)
        self.acf_days = config.get('Plots', 'acf_days', fallback=Config.Default.acf_days)

        # [Other]
        self.debug_level = config.get('Other','debug_level', fallback=Config.Default.debug_level)
        self.save_path = config.get('Other','save_path', fallback=Config.Default.save_path)

    def validate(self):
        def trim_quotes(val):
            if type(val) is not list:
                if '[' in val:
                    val = json.loads(val.replace('\'', '"'))
                    val = [i.replace('"','') for i in val]
                else:
                    val = val.replace('\'', '').replace('"', '')
            return val
        
        # Convert strings to list if needed for list values
        self.data_source = trim_quotes(self.data_source)
        self.numeric_cols = trim_quotes(self.numeric_cols)
        self.error_cols = trim_quotes(self.error_cols)
        
        # Trim quotes from string values
        self.prediction_target = self.prediction_target.replace('\'', '').replace('"', '')
        self.save_path = self.save_path.replace('\'', '').replace('"', '')
        
        # Convert strings to numeric values
        self.validation_percent = float(self.validation_percent)

        self.epochs = int(self.epochs)
        self.batch_size = int(self.batch_size)
        self.early_stopping = int(self.early_stopping)
        self.validation_batchs = int(self.validation_batchs)
        self.learning_rate = float(self.learning_rate)

        
        self.acf_days = int(self.acf_days)

        # Convert sample period to datetime object
        if type(self.sample_period_start) is not datetime:
            self.sample_period_start = parser.parse(self.sample_period_start)
        if type(self.sample_period_end) is not datetime:
            self.sample_period_end = parser.parse(self.sample_period_end)

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
    converted_data = data[numeric_cols].apply(pd.to_numeric, downcast='float', errors='coerce')
    
    # Keep site string
    converted_data['site'] = data['site']
    converted_data['timestamp'] = converted_data.apply(lambda x: datetime(int(x['year']),int(x['month']),int(x['day'])), axis=1)

    log.info('Finished loading data with {} samples present'.format(len(data)))  
    log.info('Dropped {} missing values, {} estimated values, {} null values, and {} footer rows'.format(dropped_missing, dropped_estimated, dropped_null, dropped_footer))

    return converted_data

def plot_corralation(df, savepath='Correlation.png'):
    f = plt.figure(figsize=(19, 19))
    plt.matshow(df.corr(numeric_only=True), fignum=f.number)
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=45)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)
    plt.savefig(savepath)

def plot_diff(df, key, save_path, days=14):  
    # Orginal Series
    fig, axes = plt.subplots(3,1, figsize=(20,14), sharex=False)
    for site in df['site'].unique():
        site_df = df[df['site'] == site].copy()
        axes[0].plot('timestamp', key, data=site_df, label=site)
        try:
            plot_acf(site_df[key], ax=axes[1], lags=days, label=site)
        except:
            log.error('Error generating ACF plot for {} at site {}'.format(key, site))
        try:
            plot_pacf(site_df[key], ax=axes[2], method='ols', lags=days, label=site)
        except:
            log.error('Error generating PACF plot for {} at site {}'.format(key, site))

    axes[0].set_title('Original Series')
    axes[0].grid(visible=True, axis='y')
    axes[0].legend(title='Site', bbox_to_anchor=(1.02, 0.5), loc='upper left')
    axes[1].legend(title='Site', bbox_to_anchor=(1.02, 0.5), loc='upper left')
    axes[2].legend(title='Site', bbox_to_anchor=(1.02, 0.5), loc='upper left')
    plt.savefig(save_path)
    plt.close()

def data_analysis(data, config):
    log.info('Starting data analysis')
    # Select only the sample period data.
    sp_df = data[(data['timestamp'] >= config.sample_period_start) & (data['timestamp'] < config.sample_period_end)]

    # Site specific analysis
    for site in sp_df['site'].unique():
        log.info('Analyzing site {}'.format(site))
        site_df = sp_df[sp_df['site'] == site]

        log.debug('Creating stats csv')
        # Write stats to a csv
        statsdf = site_df.describe()
        statsdf.to_csv(os.path.join(config.save_path, '{}_stats.csv'.format(site)))

        log.debug('Plotting Corralation graph')
        # Plot Corralation graph
        if not os.path.exists(os.path.join(config.save_path, 'plots')):
            os.makedirs(os.path.join(config.save_path, 'plots'))
        plot_corralation(site_df, os.path.join(config.save_path, 'plots', 'Corr_{}.png'.format(site)))

    # Variable specfic analysis
    log.info('Creating ACF, PACF plots for each variable')
    for key in tqdm(config.numeric_cols):
        if key in ['year','month','day','timestamp']:
            continue
        plot_diff(sp_df, key, os.path.join(config.save_path, 'plots', 'Diff_{}.png'.format(key)), days=config.acf_days)
        
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
    data = load_data(config.data_source, config.numeric_cols, config.error_cols)

    # Pre model analysis
    data_analysis(data, config)
    
if __name__ == '__main__':
    main()