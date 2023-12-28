import datetime
import logging
import pandas as pd

def read_and_sanitize_csv(csv_fn, index_col=None):
    ''' Loads csv_fn into a DataFrame, then:
            - casts all columns into strings
            - treats anything following '#' as a commment
            - removes any empty rows or columns
            - removes leading/trailing whitespace from column names
            - fills any empty values with ''
            - squeezes to a Series if 1D
    '''

    df = pd.read_csv(csv_fn, dtype=str, comment='#')

    df.columns = df.columns.str.strip()

    df = df.dropna(axis='index', how='all')
    
    if not df.empty:
        df = df.dropna(axis='columns', how='all')
        
    df = df.fillna('')

    if index_col is not None:
        df = df.set_index(index_col)

    possibly_series = df.squeeze(axis='columns')

    return possibly_series

def configure_standard_logger(results_dir):
    log_fn = results_dir / f'log_{datetime.datetime.now():%y%m%d-%H%M%S}.out'

    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_fn)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s',
                                    datefmt='%y-%m-%d %H:%M:%S',
                                    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    print(f'Logging in {log_fn}')

    return logger, file_handler