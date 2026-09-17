def var_g_to_chr(var_g):
    ''' Given a var_g with NBCI-style ac, return chr-style ref name.
    TODO: need a more versatile mapping that supports e.g. mm10
    '''
    chr_symbol = int(var_g.ac.split('.')[0].split('_')[-1])
    if chr_symbol == 23:
        chr_symbol = 'X'

    return f'chr{chr_symbol}'
