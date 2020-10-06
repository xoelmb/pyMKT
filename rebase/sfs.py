import numpy as np
from numba import jit, prange
import pandas as pd


def makeSfs(data, cum):
    div = {'mi': data.mi.sum(),
           'Di': data.di.sum(),
           'm0': data.m0.sum(),
           'D0': data.d0.sum()}

    daf = {'daf': np.arange(0.025, 0.980, 0.05),
           'Pi': np.array([daf.split(';') for daf in data.daf0f], dtype=int).sum(axis=0),
           'P0': np.array([daf.split(';') for daf in data.daf4f], dtype=int).sum(axis=0)}

    with jit(nopython=True):

        if cum:
            daf['Pi'] = np.cumsum(daf['Pi'][::-1])[::-1]
            daf['P0'] = np.cumsum(daf['P0'][::-1])[::-1]

    return pd.DataFrame(daf, index=range(20)), pd.DataFrame(div, index=[0])

### DEBUGGING ###

root_dir = '/home/xoel/Escritorio/mastersthesis/'
data_dir = root_dir + 'data/'

data = pd.read_csv(data_dir + 'metaPops.tsv', sep='\t')
test = data.iloc[0:50, :]