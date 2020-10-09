import numpy as np
import multiprocessing as mp
# import pathos.multiprocessing as mp
from numba import float64, int64, jitclass


n_jobs = mp.cpu_count()
# specs_daf = [('daf', float64[::1]),
#            ('Pi', int64[:]),
#            ('P0', int64[:])]
# specs_div = [('mi', int64),
#            ('m0', int64),
#            ('Di', int64),
#            ('D0', int64)]


# @jitclass(specs_daf)
# class DAF:
#     def __init__(self, Pi, P0, daf=np.arange(0.025, 0.980, 0.05)):
#         self.daf = daf
#         self.Pi = Pi
#         self.P0 = P0


# @jitclass(specs_div)
# class DIV:
#     def __init__(self, mi, m0, Di, D0):
#         self.Di = Di
#         self.D0 = D0
#         self.mi = mi
#         self.m0 = m0


def parallel_sfs(genesets, data, tests, populations):
    my_pool = mp.Pool(n_jobs)
    pars = [(geneset, data, tests, populations) for geneset in genesets]
    results = my_pool.starmap(sfs, pars)
    my_pool.terminate()
    poldivs = tuple(item for sublist in results for item in sublist)
    return poldivs
    
    
def sfs(geneset, data, tests, populations):
    cum = True if 'aMKT' in tests else False
    poldiv = []
    
    for pop in populations:
        subdata = data[(data.id.isin(geneset[1])) &
                                           (data['pop'] == pop)]
        poldiv.append((geneset[0], pop, makeSfs(subdata, cum)))
        
    return poldiv


def makeSfs(data, cum=True):
    div = dict(mi=data.mi.sum(),
              Di=data.di.sum(),
              m0=data.m0.sum(),
              D0=data.d0.sum())

    daf = dict(Pi=np.array(data.daf0f.to_list(), dtype=int).sum(axis=0),
              P0=np.array(data.daf4f.to_list(), dtype=int).sum(axis=0))

    if cum:
        daf_cum = dict(Pi=np.cumsum(daf['Pi'][::-1])[::-1],
                      P0=np.cumsum(daf['P0'][::-1])[::-1])
        return (('daf', daf), ('daf_cum', daf_cum), ('div', div))
    else:
        return (('daf', daf), ('div', div))


### DEBUGGING ###
# import pandas as pd
# root_dir = '/home/xoel/Escritorio/mastersthesis/'
# data_dir = root_dir + 'data/'

# data = pd.read_csv(data_dir + 'metaPops.tsv', sep='\t')
# test = data.iloc[0:50, :]