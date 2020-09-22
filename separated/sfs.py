#!/usr/bin/env python

import numpy as np
import pandas as pd

def makeSfs(data, cum):
    div = {'mi': data.mi.sum(),
           'Di': data.di.sum(),
           'm0': data.m0.sum(),
           'D0': data.d0.sum()}

    daf = {'daf': np.arange(0.025, 0.980, 0.05),
           'Pi': np.array(tuple(map(sum, zip(*tuple([tuple(map(int, daf.split(';'))) for daf in data.daf0f]))))),
           'P0': np.array(tuple(map(sum, zip(*tuple([tuple(map(int, daf.split(';'))) for daf in data.daf4f])))))}

    if cum:
        daf['Pi'] = np.cumsum(daf['Pi'][::-1])[::-1]
        daf['P0'] = np.cumsum(daf['P0'][::-1])[::-1]

    return pd.DataFrame(daf, index=range(20)), pd.DataFrame(div, index=[0])

