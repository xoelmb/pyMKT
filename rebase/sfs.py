import numpy as np
import multiprocessing as mp
# import pathos.multiprocessing as mp
# from numba import float64, int64, jitclass, unicode_type
from numba import types, typed, jit

n_jobs = mp.cpu_count()

def parallel_sfs(genesets, data, tests, thresholds, populations):
    my_pool = mp.Pool(n_jobs)
    pars = [(geneset, data, tests, thresholds, populations) for geneset in genesets]
    results = my_pool.starmap(sfs, pars)
    my_pool.terminate()
    poldivs = tuple(item for sublist in results for item in sublist)
    return poldivs
    
    
def sfs(geneset, data, tests, thresholds, populations):

    cum = True if 'aMKT' in tests else False
    poldivs = []
    
    for pop in populations:
        subdata = data[(data.id.isin(geneset[1])) &
                       (data['pop'] == pop)]
        pd = makeSfs(subdata, cum)

        for t, ths in zip(tests, thresholds):
            use_daf = 'daf_cum' if t == 'aMKT' else 'daf'
            for th in ths:
                poldivs.append(dict(name=geneset[0],
                                    population=pop,
                                    daf=pd[use_daf],
                                    div=pd['div'],
                                    test=t,
                                    threshold=th))

    return poldivs


def makeSfs(data, cum=True):

    div = dict(mi=data.mi.sum(),
              Di=data.di.sum(),
              m0=data.m0.sum(),
              D0=data.d0.sum())

    daf = dict(Pi=np.array(data.daf0f.to_list(), dtype=int).sum(axis=0),
              P0=np.array(data.daf4f.to_list(), dtype=int).sum(axis=0))

    if cum:
        daf_cum = dict(Pi=np.ascontiguousarray(np.cumsum(daf['Pi'][::-1])[::-1]),
                      P0=np.ascontiguousarray(np.cumsum(daf['P0'][::-1])[::-1]))
        return {'daf': daf, 'daf_cum': daf_cum, 'div': div}
    
    else:
        return {'daf': daf, 'div': div}
