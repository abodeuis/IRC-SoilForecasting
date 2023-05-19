import os
import logging
import argparse

import src.logging_utils as log_utils
import src.analysis as analysis
from src.config import Config
import src.dataloader as icn_data

log = logging.getLogger('ICN-Forecasting')

def parse_command_line():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-c', '--config', default='config.ini', action='store', help='Config file to load from')

    args = parser.parse_args()
    return args

def main():
    args = parse_command_line()

    # Start logger
    log_utils.setup_logger('Latest.log', logging.DEBUG)

    # Load User Config
    config = Config(args.config)

    if config.save_path != '' and not os.path.exists(os.path.join(config.save_path, 'logs')):
        os.makedirs(os.path.join(config.save_path, 'logs'))
    log_utils.set_log_file(os.path.join(config.save_path, 'logs', 'analysis.log'))
    log_utils.set_log_level(config.debug_level)

    # Load Dataset
    data = icn_data.load_data(config.data_source, config.numeric_cols, config.error_cols)

    # Pre model analysis
    analysis.data_analysis(data, config)
    
if __name__ == '__main__':
    main()