import os
import json
import logging
import configparser

from dateutil import parser
from datetime import datetime

log = logging.getLogger('ICN-Forecasting')

class Config:
    class Default:
        # [Data]
        data_source = ''
        target_cols = 'avg_soiltemp_4in_sod' # Name of the column that we are going to try to predict
        validation_percent = 0.2 # Percentage of training set to use as validation
        numeric_cols = ['year', 'month', 'day', 'max_wind_gust', 'avg_wind_speed', 'avg_wind_dir', 'sol_rad', 'max_air_temp', 'min_air_temp', 'avg_air_temp', 'max_rel_hum', 'min_rel_hum', 'avg_rel_hum', 'avg_dewpt_temp', 'precip', 'pot_evapot', 'max_soiltemp_4in_sod', 'min_soiltemp_4in_sod', 'avg_soiltemp_4in_sod', 'max_soiltemp_8in_sod', 'min_soiltemp_8in_sod', 'avg_soiltemp_8in_sod', 'max_soiltemp_4in_bare', 'min_soiltemp_4in_bare', 'avg_soiltemp_4in_bare', 'max_soiltemp_2in_bare', 'min_soiltemp_2in_bare', 'avg_soiltemp_2in_bare', 'SM5', 'SM10', 'SM20', 'SM50', 'SM100', 'SM150']
        error_cols = ['xwser', 'awser', 'awder', 'soler', 'xater', 'nater', 'aater', 'xrher', 'nrher', 'arher', 'adper', 'pcer', 'pevaper', 'xst4soder', 'nst4soder', 'ast4soder', 'xst8soder', 'nst8soder', 'ast8soder', 'xst4bareer', 'nst4bareer', 'ast4bareer', 'xst2bareer', 'nst2bareer', 'ast2bareer']

        # [Training]
        training_win_size = 14    # Number of time steps to use for each training step window
        prediction_win_size = 7   # Number of time steps to predict out.
        epochs = 50          # Number of training epochs
        batch_size = 32     # Number of individual samples to use for each training step.
        early_stopping = 5  # Early stopping, will stop training if no improvement in 'N' epochs
        validation_batchs = 3 # Number of batchs to use per training step as validation
        learning_rate = 1e-3 # learning rate

        # [Plots]
        sample_period_start = datetime(1900,1,1)
        sample_period_end = datetime(2100,1,1)
        acf_days = 14       # Amount of days to forecast out for the ACF PACF plots

        # [Other]
        debug_level = 'INFO'
        save_path = os.path.join('training_runs', datetime.now().strftime('model_%d%m%Y_%H%M%S'))# Path to save the model to

        @staticmethod
        def write(file):
            content = ("# This is the configuration file for simple_rnn.py.\n" +
                       "\n" +
                       "[Data]\n" +
                       "# The directory or files to load the data from\n" +
                       "data_source = \n"
                       "# Name of the column that is going to be predicted\n" +
                       "target_cols = \'{}\'\n".format(Config.Default.target_cols) +
                       "# Percentage of the data to use for validation\n" +
                       "validation_percent = {}\n".format(Config.Default.validation_percent) +
                       "# The columns that will be used for analysis\n" +
                       "numeric_cols = {}\n".format(Config.Default.numeric_cols) +
                       "# The columns that will be used for flagig error data should contain a 'E' or 'M'\n" +
                       "error_cols = {}\n".format(Config.Default.error_cols) +
                       "\n" +
                       "[Training]\n" +
                       "# Number of time steps to use for each training step window\n" +
                       "training_win_size = {}\n".format(Config.Default.training_win_size) +
                       "# Number of time steps to predict out.\n" +
                       "prediction_win_size = {}\n".format(Config.Default.prediction_win_size) +
                       "# Number of epochs to train for.\n" +
                       "epochs = {}\n".format(Config.Default.epochs) +
                       "batch_size = {}\n".format(Config.Default.batch_size) +
                       "early_stopping = {}\n".format(Config.Default.early_stopping) +
                       "validation_batchs = {}\n".format(Config.Default.validation_batchs) +
                       "learning_rate = {}\n".format(Config.Default.learning_rate) + 
                       "\n" +
                       "[Plots]\n" +
                       "# Start Time of the sample period. Format is (Month/Day/Year Hour:Min:Sec)\n" +
                       "sample_period_start = \'{}\'\n".format(Config.Default.sample_period_start.strftime('%m/%d/%Y %H:%M:%S')) +
                       "# End Time of the sample period\n" +
                       "sample_period_end = \'{}\'\n".format(Config.Default.sample_period_end.strftime('%m/%d/%Y %H:%M:%S')) +
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
        self.target_cols = config.get('Data','target_cols', fallback=Config.Default.target_cols)
        self.validation_percent = config.get('Data','validation_percent', fallback=Config.Default.validation_percent)
        self.numeric_cols = config.get('Data','numeric_cols', fallback=Config.Default.numeric_cols)
        self.error_cols = config.get('Data', 'error_cols', fallback=Config.Default.error_cols)

        # [Training]
        self.training_win_size = config.get('Training','training_win_size', fallback=Config.Default.training_win_size)
        self.prediction_win_size = config.get('Training','prediction_win_size', fallback=Config.Default.prediction_win_size)
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
        
        # Required options check.
        if self.data_source == '':
             log.critical('No data source was given. Cannot run without any data')
             exit(1)

        # Convert strings to list if needed for list values
        self.data_source = trim_quotes(self.data_source)
        self.numeric_cols = trim_quotes(self.numeric_cols)
        self.error_cols = trim_quotes(self.error_cols)
        
        # Trim quotes from string values
        self.target_cols = self.target_cols.replace('\'', '').replace('"', '')
        self.save_path = self.save_path.replace('\'', '').replace('"', '')
        if self.save_path == '':
            self.save_path = Config.Default.save_path

        # Convert strings to numeric values
        self.validation_percent = float(self.validation_percent)

        self.training_win_size = int(self.training_win_size)
        self.prediction_win_size = int(self.prediction_win_size)
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

class TrainedConfig:
    class Default:
        # [Models]
        model_files = ''

        # This should be generated with the model save files and not adjusted
        # [Training Data]
        training_data_source = ''
        training_start_date = datetime(1900,1,1)
        training_end_date = datetime(2100,1,1)
        target_cols = 'avg_soiltemp_4in_sod' # Name of the column that we are going to try to predict
        numeric_cols = ['year', 'month', 'day', 'max_wind_gust', 'avg_wind_speed', 'avg_wind_dir', 'sol_rad', 'max_air_temp', 'min_air_temp', 'avg_air_temp', 'max_rel_hum', 'min_rel_hum', 'avg_rel_hum', 'avg_dewpt_temp', 'precip', 'pot_evapot', 'max_soiltemp_4in_sod', 'min_soiltemp_4in_sod', 'avg_soiltemp_4in_sod', 'max_soiltemp_8in_sod', 'min_soiltemp_8in_sod', 'avg_soiltemp_8in_sod', 'max_soiltemp_4in_bare', 'min_soiltemp_4in_bare', 'avg_soiltemp_4in_bare', 'max_soiltemp_2in_bare', 'min_soiltemp_2in_bare', 'avg_soiltemp_2in_bare', 'SM5', 'SM10', 'SM20', 'SM50', 'SM100', 'SM150']
        error_cols = ['xwser', 'awser', 'awder', 'soler', 'xater', 'nater', 'aater', 'xrher', 'nrher', 'arher', 'adper', 'pcer', 'pevaper', 'xst4soder', 'nst4soder', 'ast4soder', 'xst8soder', 'nst8soder', 'ast8soder', 'xst4bareer', 'nst4bareer', 'ast4bareer', 'xst2bareer', 'nst2bareer', 'ast2bareer']

        # [Prediction Data]
        prediction_data_source = ''
        prediction_start_date = datetime(1900,1,1)
        prediction_end_date = datetime(2100,1,1)

        # [Plots]
        prediction_period_start = datetime(1900,1,1)
        prediction_period_end = datetime(2100,1,1)

        # [Other]
        debug_level = 'INFO'
        save_path =  os.path.join('predictions', datetime.now().strftime('prediction_%d%m%Y_%H%M%S')) # Path to save the model to
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.load(filepath)
        self.validate()

    def load(self, filepath):
        log.info('Loading prediction config from \"{}\"'.format(filepath))
        if not os.path.isfile(filepath):
            log.warning('No config file found at \"{}\". Using defaults instead'.format(filepath))
            
        config = configparser.ConfigParser()
        config.read(filepath)

        # [Data]
        self.data_source = config.get('Data','data_source', fallback=Config.Default.data_source)
        self.target_cols = config.get('Data','target_cols', fallback=Config.Default.target_cols)
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
        
        # Required options check.
        if self.data_source == '':
             log.critical('No data source was given. Cannot run without any data')
             exit(1)

        # Convert strings to list if needed for list values
        self.data_source = trim_quotes(self.data_source)
        self.numeric_cols = trim_quotes(self.numeric_cols)
        self.error_cols = trim_quotes(self.error_cols)
        
        # Trim quotes from string values
        self.target_cols = self.target_cols.replace('\'', '').replace('"', '')
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