from eMKT import *
from process_parallel import *
import pandas as pd
import numpy as np

def makeSfs(data, cum):
    daf = {}
    div = {}

    daf['f'] = np.arange(0.025, 0.980, 0.05)
    daf['pi'] = np.array(tuple(map(sum, zip(*tuple([tuple(map(int, daf.split(';'))) for daf in data.daf0f])))))
    daf['p0'] = np.array(tuple(map(sum, zip(*tuple([tuple(map(int, daf.split(';'))) for daf in data.daf4f])))))

    if cum:
        daf['pi'] = np.cumsum(pi[::-1])[::-1]
        daf['p0'] = np.cumsum(p0[::-1])[::-1]

    div['mi'], div['di'], div['m0'], div['d0'] = data.mi.sum(), data.di.sum(), data.m0.sum(), data.d0.sum()

    return daf, div

def mkt_on_df(gene_df, data_df, label=None, pops=None, tests=None, cutoffs=None, do_trims=None, bootstrap=None,
              b_size=None, b_reps=None):
    if do_trims is None:
        do_trims = [True, False]
    if cutoffs is None:
        cutoffs = [0.05, 0.15]
    if tests is None:
        tests = ['eMKT', 'aMKT']
    if pops is None:
        pops = ['AFR', 'EUR']
    if bootstrap is None:
        bootstrap = False
    if b_reps is None:
        b_reps = 100

    pars = [(gene_df.iloc[:, i], data_df, pops, tests, cutoffs, do_trims, bootstrap, b_size, b_reps) for i in
            range(len(gene_df.columns.values))]

    # Loads the models for all the parameters parsed using multiprocessing to speed up computations
    pool = MyPool(processes=8)  # multiprocessing.cpu_count())
    results_list = pool.starmap(mkt_on_col, pars)
    pool.terminate()
    results = pd.concat(results_list, axis=0, ignore_index=True)

    if label is not None: results['label'] = label

    return results


def mkt_on_col(col, data_df, pops=None, tests=None, cutoffs=None, do_trims=None, bootstrap=None, b_size=None,
               b_reps=None):
    if do_trims is None:
        do_trims = [True, False]
    if cutoffs is None:
        cutoffs = [0.05, 0.15]
    if tests is None:
        tests = ['eMKT', 'aMKT']
    if pops is None:
        pops = ['AFR', 'EUR']
    if bootstrap is None:
        bootstrap = False
    if b_reps is None:
        b_reps = 100

    glist = col[col == 1].index.values
    if len(glist) > 0:
        results = mkt_on_list(glist, data_df, pops, tests, cutoffs, do_trims, bootstrap, b_size, b_reps)
    else:
        results = pd.DataFrame(index=[0])
        results['nogenes'] = 0

    if col.name is not None:
        results['stage'] = col.name[0]
        results['region'] = col.name[1]
    return results


def mkt_on_list(glist, data_df, pops=None, tests=None, cutoffs=None, do_trims=None, bootstrap=None, b_size=None,
                b_reps=None):
    if do_trims is None:
        do_trims = [True, False]
    if cutoffs is None:
        cutoffs = [0.05, 0.15]
    if tests is None:
        tests = ['eMKT', 'aMKT']
    if pops is None:
        pops = ['AFR', 'EUR']
    if bootstrap is None:
        bootstrap = False
    if b_reps is None:
        b_reps = 100

    df = data_df[data_df['id'].isin(glist)]

    pars = []
    for pop in pops:
        subdata = df[df['pop'] == pop]
        if bootstrap:
            pars.append((subdata, pop, tests, cutoffs, do_trims, b_size, b_reps))
        else:
            pars.append((subdata, pop, tests, cutoffs, do_trims))

    func = bootstrap_on_subdata if bootstrap else mkt_on_subdata
    # Loads the models for all the parameters parsed using multiprocessing to speed up computations
    pool = MyPool(processes=8)  # multiprocessing.cpu_count())
    results_list = pool.starmap(func, pars)
    pool.terminate()
    results = pd.concat(results_list, axis=0, ignore_index=True)

    return results


def bootstrap_on_subdata(subdata, pop=None, tests=None, cutoffs=None, do_trims=None, b_size=None, b_reps=None):
    nogenes = len(subdata.index.values)
    if nogenes <= 0:
        results = pd.DataFrame(index=[0])
        results['nogenes'] = 0
    else:
        if do_trims is None:
            do_trims = [True, False]
        if cutoffs is None:
            cutoffs = [0.05, 0.15]
        if tests is None:
            tests = ['eMKT', 'aMKT']
        if b_size is None:
            b_size = nogenes
        if b_reps is None:
            b_reps = 100

        pars = [(subdata.sample(n=b_size, replace=True), pop, tests, cutoffs, do_trims) for _ in range(b_reps)]
        pool = MyPool(processes=2)  # multiprocessing.cpu_count())
        results_list = pool.starmap(mkt_on_subdata, pars)
        pool.terminate()

        results = pd.concat(results_list, axis=0, ignore_index=True)

    return results


def mkt_on_subdata(subdata, pop=None, tests=None, cutoffs=None, do_trims=None):
    if do_trims is None:
        do_trims = [True, False]
    if cutoffs is None:
        cutoffs = [0.05, 0.15]
    if tests is None:
        tests = ['eMKT', 'aMKT']

    nogenes = len(subdata.index.values)
    if nogenes <= 0:
        results = pd.DataFrame(index=[0])
    else:
        if 'aMKT' in tests:
            daf_cum, div = makeSfs(subdata, cum=True)
        if 'eMKT' in tests:
            daf, div = makeSfs(subdata, cum=False)

        pars = []
        for test in tests:
            if test == 'eMKT':
                for cutoff in cutoffs:
                    pars.append([daf, div, test, cutoff])
            elif test == 'aMKT':
                for do_trim in do_trims:
                    pars.append((daf_cum, div, test, do_trim))

        # Loads the models for all the parameters parsed using multiprocessing to speed up computations
        pool = MyPool(processes=2)  # multiprocessing.cpu_count())
        results_list = pool.starmap(mkt_on_daf, pars)
        pool.terminate()

        results = pd.concat(results_list, axis=0, ignore_index=True)

    if pop is not None: results['pop'] = pop
    results['nogenes'] = nogenes

    return results


def mkt_on_daf(daf, div, test, par):
    try:
        if test == 'eMKT':
            results = eMKT(daf, div, par)
            results = pd.DataFrame(results, index=[0])
            label_col = 'cutoff'
        elif test == 'aMKT':
            if par:
                xlow = 0.1
                xhigh = 0.9
            else:
                xlow = 0
                xhigh = 1
            results = aMKT(daf, div, xlow, xhigh, check='raise')
            if 'alpha' not in results.keys():  # Uses covar matrix even if it's not valid, leads to bad CI's
                results = aMKT(daf, div, xlow, xhigh, check='ignore')

            results = pd.DataFrame(results, index=[0])
            label_col = 'trim'
        else:
            return None
    except:
        results = pd.DataFrame(index=[0])
        label_col = 'cutoff' if test == 'eMKT' else 'trim'

    results['test'] = test
    results[label_col] = par

    return results
