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