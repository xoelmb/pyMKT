import numpy as np
import multiprocessing as mp
import functools
import sys

n_jobs = mp.cpu_count()


def parallel_sfs(genesets, popdata, tests, thresholds, bootstrap=False, reps=100, v=True, c=25):
    
    mypool = mp.Pool(n_jobs)
    
    func = functools.partial(sfs, popdata=popdata, tests=tests, thresholds=thresholds, bootstrap=bootstrap, reps=reps)

    if v:
        n=len(genesets)
        results = []
        for i, r in enumerate(mypool.imap_unordered(func, genesets, chunksize=25), 1):
            results.append(r)
            sys.stderr.write(f'\r· [1/2] Parsing {"and bootstrapping " if bootstrap else ""}population data {round(i/n*100,2)}%')
        sys.stderr.write(f'\r· [1/2] Parsing {"and bootstrapping " if bootstrap else ""}population data [DONE]\n')
    else:
        results = list(mypool.imap_unordered(func, genesets, chunksize=30))

    mypool.terminate()
    poldivs = tuple(item for sublist in results for item in sublist)
    
    return poldivs
    
    
def sfs(geneset, popdata, tests, thresholds, bootstrap=False, reps=100):

    cum = True if 'aMKT' in tests else False
    poldivs = []
    
    for pop, data in popdata.items():
        data = data[data.id.isin(geneset[1])]

        for _ in range(reps if bootstrap else 1):
            idata = data.sample(len(data), replace=True) if bootstrap else data
            poldiv_i = makeSfs(idata, cum)

            for t, ths in zip(tests, thresholds):
                use_daf = 'daf_cum' if t == 'aMKT' else 'daf'
                for th in ths:
                    poldivs.append(dict(name=geneset[0],
                                        population=pop,
                                        daf=poldiv_i[use_daf],
                                        div=poldiv_i['div'],
                                        test=t,
                                        threshold=th,
                                        ngenes=len(data.index)))
    return poldivs


def makeSfs(data, cum=True):

    div = dict(mi=data['mi'].sum(),
              Di=data['di'].sum(),
              m0=data['m0'].sum(),
              D0=data['d0'].sum())

    daf = dict(Pi=data[[f'Pi{i}' for i in range(20)]].sum().to_numpy(),
               P0=data[[f'P0{i}' for i in range(20)]].sum().to_numpy())

    if cum:
        daf_cum = dict(Pi=np.cumsum(daf['Pi'][::-1])[::-1],
                      P0=np.cumsum(daf['P0'][::-1])[::-1])
        return {'daf': daf, 'daf_cum': daf_cum, 'div': div}

    return {'daf': daf, 'div': div}

