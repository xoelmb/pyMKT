import numpy as np
import multiprocessing as mp
import functools
import sys
import pandas as pd

n_jobs = mp.cpu_count()


def parallel_sfs(genesets, popdata, tests, thresholds, permute=False, bootstrap=False, reps=100, v=True, permute_vars_alone=False, permute_vars_and_constant=True, c=25):
    
    mypool = mp.Pool(n_jobs)
    
    func = functools.partial(sfs, popdata=popdata, tests=tests, thresholds=thresholds, permute=permute, bootstrap=bootstrap, reps=reps, permute_vars_alone=permute_vars_alone, permute_vars_and_constant=permute_vars_and_constant)

    if v:
        sys.stderr.write(f'· [1/2] Parsing {"and resampling " if (bootstrap or permute) else ""}population data ...')
        n=len(genesets)
        results = []
        for i, r in enumerate(mypool.imap_unordered(func, genesets, chunksize=25), 1):
            results.append(r)
            sys.stderr.write(f'\r· [1/2] Parsing {"and resampling " if (bootstrap or permute) else ""}population data {round(i/n*100,2)}%')
        sys.stderr.write(f'\r· [1/2] Parsing {"and resampling " if (bootstrap or permute) else ""}population data [DONE]\n')
    else:
        results = list(mypool.imap_unordered(func, genesets, chunksize=30))

    mypool.terminate()
    poldivs = tuple(item for sublist in results for item in sublist)
    
    return poldivs
    
    
def sfs(geneset, popdata, tests, thresholds, permute=False, bootstrap=False, reps=100, permute_vars_alone=False, permute_vars_and_constant=True):

    def aux_single():
        return [makeSfs(data, cum)], [len(data.index)], ['single']

    def aux_bootstrap():
        idata = data.sample(len(data), replace=True)
        return [makeSfs(idata, cum)], [len(idata.index)], ['bootstrap']
    
    def aux_permute():

        idata_var = data[data.id.isin(geneset[2])].sample(N_VARS, replace=False)
        poldiv_is = []
        ngenes = []
        repeats = []

        if permute_vars_alone:
            poldiv_is.append(makeSfs(idata_var, cum))
            ngenes.append(len(idata_var.index))
            repeats.append('perm_vars_alone')
            
        if permute_vars_and_constant:
            
            idata_var_ct = pd.concat([idata_var, data_constant], axis=0)
            
            poldiv_is.append(makeSfs(idata_var_ct, cum))
            ngenes.append(len(idata_var_ct.index))
            repeats.append('perm_vars+ct')
        
        return poldiv_is, ngenes, repeats


    cum = True if 'aMKT' in tests else False
    poldivs = []
    
    mode_function = aux_bootstrap if bootstrap else aux_permute if permute else aux_single



    for pop, data in popdata.items():
        data = data[data.id.isin(geneset[1])]
        
        if permute:
            # Store constant genes
            CONSTANT_set = set(geneset[1])-set(geneset[2])
            data_constant = data[data.id.isin(CONSTANT_set)]
            # Count variable genes
            N_VARS = len(set(geneset[1])-CONSTANT_set)

        for _ in range(reps if (bootstrap or permute) else 1):
            
            poldiv_is, ngenes, repeats = mode_function()
                
            for poldiv_i, ng, repeat in zip(poldiv_is, ngenes, repeats):
                for t, ths in zip(tests, thresholds):
                    use_daf = 'daf_cum' if t == 'aMKT' else 'daf'
                    for th in ths:
                        poldivs.append(dict(name=geneset[0],
                                            population=pop,
                                            daf=poldiv_i[use_daf],
                                            div=poldiv_i['div'],
                                            test=t,
                                            threshold=th,
                                            ngenes=ng, 
                                            repeat=repeat))
    # for pop, data in popdata.items():
    #     data = data[data.id.isin(geneset[1])]

    #     for _ in range(reps if (bootstrap or permute) else 1):
    #         if bootstrap:
    #             idata = data.sample(len(data), replace=True)
            # else:
            #     idata = data
            
            # poldiv_i = makeSfs(idata, cum)

            # for t, ths in zip(tests, thresholds):
            #     use_daf = 'daf_cum' if t == 'aMKT' else 'daf'
            #     for th in ths:
            #         poldivs.append(dict(name=geneset[0],
            #                             population=pop,
            #                             daf=poldiv_i[use_daf],
            #                             div=poldiv_i['div'],
            #                             test=t,
            #                             threshold=th,
            #                             ngenes=len(idata.index)))
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

