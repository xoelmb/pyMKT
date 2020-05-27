#!/usr/bin/env python

import pandas as pd
import numpy as np
from fisher import pvalue
from scipy import optimize
import multiprocessing
import multiprocessing.pool
import copy

from numba import jit


class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess


# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(MyPool, self).__init__(*args, **kwargs)


def makeSfs(x, cum=False):
    f = np.arange(0.025, 0.980, 0.05)
    pi = pd.DataFrame(x['daf0f'].apply(lambda daf0f: [int(num) for num in daf0f.split(';')]).to_list()).sum()
    p0 = pd.DataFrame(x['daf4f'].apply(lambda daf4f: [int(num) for num in daf4f.split(';')]).to_list()).sum()

    daf = pd.concat([pd.Series(f), pi, p0], axis='columns', ignore_index=True)
    daf.columns = ['daf', 'Pi', 'P0']
    div = pd.DataFrame(x[['mi', 'di', 'm0', 'd0']].sum(), dtype=int).transpose().rename(
        columns={'di': 'Di', 'd0': 'D0'})

    if cum:
        daf = cumulative(daf)

    return daf, div


def cumulative(x):
    psyn = [x['P0'].sum()] + [0] * (len(x) - 1)
    pnsyn = [x['Pi'].sum()] + [0] * (len(x) - 1)
    for i in range(1, len(x)):
        appS = psyn[i - 1] - x['P0'][i - 1]
        appNsyn = pnsyn[i - 1] - x['Pi'][i - 1]
        if (appS > 0) & (appNsyn > 0):
            psyn[i] = appS
            pnsyn[i] = appNsyn
        else:
            psyn[i] = 0
            pnsyn[i] = 0

    x['P0'] = psyn
    x['Pi'] = pnsyn

    return x


# @jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def eMKT(daf, div, cutoff=0.15):
    res = {}

    P0 = daf['P0'].sum()
    Pi = daf['Pi'].sum()
    D0 = int(div['D0'])
    Di = int(div['Di'])
    m0 = int(div['m0'])
    mi = int(div['mi'])

    # Divergence metrics
    res['Ka'] = Di / mi
    res['Ks'] = D0 / m0
    res['omega'] = res['Ka'] / res['Ks']

    ### Estimating alpha with Pi/P0 ratio
    PiMinus = daf[daf['daf'] <= cutoff]['Pi'].sum()
    PiGreater = daf[daf['daf'] > cutoff]['Pi'].sum()
    P0Minus = daf[daf['daf'] <= cutoff]['P0'].sum()
    P0Greater = daf[daf['daf'] > cutoff]['P0'].sum()

    ratioP0 = P0Minus / P0Greater
    deleterious = PiMinus - (PiGreater * ratioP0)
    PiNeutral = Pi - deleterious

    res['alpha'] = 1 - (((Pi - deleterious) / P0) * (D0 / Di))

    ## Estimation of b: weakly deleterious
    res['neg_b'] = (deleterious / P0) * (m0 / mi)

    ## Estimation of f: neutral sites
    res['neg_f'] = (m0 * PiNeutral) / (mi * P0)

    ## Estimation of d, strongly deleterious sites
    res['neg_d'] = 1 - (res['neg_f'] + res['neg_b'])

    res['pvalue'] = pvalue(P0, D0, Pi - deleterious, Di).two_tail

    ## Omega A and Omega D
    res['omegaA'] = res['omega'] * res['alpha']
    res['omegaD'] = res['omega'] - res['omegaA']

    return res


def aMKT(daf, div, xlow=0, xhigh=1):
    res = {}

    P0 = daf['P0'].sum()
    Pi = daf['Pi'].sum()
    D0 = int(div['D0'])
    Di = int(div['Di'])
    m0 = int(div['m0'])
    mi = int(div['mi'])

    ### Divergence metrics
    res['Ka'] = Di / mi
    res['Ks'] = D0 / m0
    res['omega'] = res['Ka'] / res['Ks']

    ## Estimate the synonymous and non-synonymous ratio
    synonymousRatio = P0 / m0
    nonSynonymousRatio = Pi / mi

    ## Estimate the fraction of neutral sites incluiding weakly deleterious variants
    fb = nonSynonymousRatio / synonymousRatio

    ## Estimate the fraction of strongly deleleterious sites (d)
    res['neg_d'] = 1 - fb

    try:
        ## Run asymptotic MKT and retrieve alphas
        model = amkt_fit(daf, div, xlow, xhigh)
        res.update(model)

    except:
        #         print(e)
        daf10 = daf.copy(deep=True)
        daf10['daf'] = np.array([[x / 100, x / 100] for x in range(5, 100, 10)]).flatten()
        daf10 = daf.groupby('daf', as_index=False).sum()

        try:
            model = amkt_fit(daf, div, xlow, xhigh)
            res.update(model)
            res['daf10'] = True

        except:
            return res

    # Estimate the fraction of sligthly deleterious sites in each daf category (b)
    omegaD = daf['Pi'] - (((1 - res['alpha']) * Di * daf['P0']) / D0)
    res['neg_b'] = (omegaD.sum() / daf['P0'].sum()) * (m0 / mi)

    # Re-estimate the truly number of neutral sites, removing the slightly deleterious 
    res['neg_f'] = fb - res['neg_b']

    ## Omega A and Omega
    res['omegaA'] = res['omega'] * res['alpha']
    res['omegaD'] = res['omega'] - res['omegaA']

    return res


def amkt_fit(daf, div, xlow, xhigh):
    res = {}

    d_ratio = float(div['D0'] / div['Di'])

    # Compute alpha values and trim
    alpha = 1 - d_ratio * (daf['Pi'] / daf['P0'])
    trim = ((daf['daf'] >= xlow) & (daf['daf'] <= xhigh))

    # Two-step nls2() model fit at a given level of precision (res)
    try:
        model = optimize.curve_fit(exp_model, daf['daf'][trim], alpha[trim], method='lm')
    #         print('Fit: lm')
    except:
        try:
            model = optimize.curve_fit(exp_model, daf['daf'][trim], alpha[trim], method='trf')
        #             print('Fit: trf')
        except:
            try:
                model = optimize.curve_fit(exp_model, daf['daf'][trim], alpha[trim], method='dogbox')
            #                 print('Fit: dogbox')
            except:
                raise RuntimeError("Couldn't fit any method")

    res['a'] = model[0][0]
    res['b'] = model[0][1]
    res['c'] = model[0][2]

    # alpha for predicted model
    res['alpha'] = exp_model(1.0, res['a'], res['b'], res['c'])

    # Compute confidence intervals based on simulated data (MC-SOERP)
    vcov = pd.concat([pd.DataFrame([0] * 4).transpose(),
                      pd.concat([pd.DataFrame([0] * 4), pd.DataFrame(model[1])], axis=1, ignore_index=True)],
                     axis=0, ignore_index=True)
    vcov = vcov.iloc[0:4, :].values

    simpars = np.random.multivariate_normal(mean=[1.0, res['a'], res['b'], res['c']], cov=vcov, size=10000,
                                            check_valid='ignore')

    res['ciLow'], res['ciHigh'] = np.quantile([exp_model(x[0], x[1], x[2], x[3]) for x in simpars], [0.025, 0.975])

    return res


@jit(nopython=True)  # Set "nopython" mode for best performance, equivalent to @njit
def exp_model(f_trimmed, a, b, c):
    return a + b * np.exp(-c * f_trimmed)


def mkt_on_df(gene_df, data_df, approach=None, pops=['AFR', 'EUR'], tests=['eMKT', 'aMKT'], cutoffs=[0.05, 0.15],
              do_trims=[True, False]):
    pars = [(gene_df.iloc[:, i], data_df, pops, tests, cutoffs, do_trims) for i in range(len(gene_df.columns.values))]

    # Loads the models for all the parameters parsed using multiprocessing to speed up computations
    pool = MyPool(processes=8)  # multiprocessing.cpu_count())
    results_list = pool.starmap(mkt_on_col, pars)
    pool.terminate()
    results = pd.concat(results_list, axis=0, ignore_index=True)

    if approach is not None: results['approach'] = approach

    return results


def mkt_on_col(col, data_df, pops=['AFR', 'EUR'], tests=['eMKT', 'aMKT'], cutoffs=[0.05, 0.15], do_trims=[True, False]):
    # glists = {'+': col[col == 1].index.values, '-': col[col == 0].index.values}
    # pars = [(glists[gtype], data_df, gtype, pops, tests, cutoffs, do_trims) for gtype in glists.keys()]

    # pool = MyPool(processes=2)  # multiprocessing.cpu_count())
    # results_list = copy.deepcopy(pool.starmap(mkt_on_list, pars))
    # pool.terminate()
    # results = pd.concat(results_list, axis=0, ignore_index=True)

    results = mkt_on_list(col[col == 1].index.values, data_df, pops, tests, cutoffs, do_trims)

    if col.name is not None:
        results['stage'] = col.name[0]
        results['region'] = col.name[1]
        # print(col.name, 'done')
    return results


def mkt_on_list(glist, data_df, pops=['AFR', 'EUR'], tests=['eMKT', 'aMKT'], cutoffs=[0.05, 0.15],
                do_trims=[True, False]):
    df = data_df[data_df['id'].isin(glist)]
    dafs = {}
    divs = {}
    dafs_cum = {}
    nogenes = {}

    for pop in pops:
        pop_df = df[df['pop'] == pop]
        nogenes[pop] = len(pop_df.index.values)

        if 'aMKT' in tests:
            dafs_cum[pop], divs[pop] = makeSfs(pop_df, cum=True)
        if 'eMKT' in tests:
            dafs[pop], divs[pop] = makeSfs(pop_df, cum=False)

    pars = []
    for pop in pops:
        for test in tests:
            if test == 'eMKT':
                for cutoff in cutoffs:
                    pars.append([dafs[pop], divs[pop], pop, nogenes[pop], test, cutoff])
            elif test == 'aMKT':
                for do_trim in do_trims:
                    pars.append((dafs_cum[pop], divs[pop], pop, nogenes[pop], test, do_trim))

    # Loads the models for all the parameters parsed using multiprocessing to speed up computations
    pool = MyPool(processes=2)  # multiprocessing.cpu_count())
    results_list = copy.deepcopy(pool.starmap(mkt_on_daf, pars))
    pool.terminate()

    results = pd.concat(results_list, axis=0, ignore_index=True)

    # if gtype is not None: results['gtype'] = gtype

    return results


def mkt_on_daf(daf, div, pop, nogenes, test, par):
    try:
        if test == 'eMKT':
            results = copy.deepcopy(eMKT(daf, div, par))
            results = pd.DataFrame(results, index=[0])
            label_col = 'cutoff'
        elif test == 'aMKT':
            if par:
                xlow = 0.1
                xhigh = 0.9
            else:
                xlow = 0
                xhigh = 1
            results = copy.deepcopy(aMKT(daf, div, xlow, xhigh))
            results = pd.DataFrame(results, index=[0])
            label_col = 'trim'
        else:
            return None
    except:
        results = pd.DataFrame(index=[0])
        label_col = 'cutoff' if test == 'eMKT' else 'trim'

    if pop is not None: results['pop'] = pop
    if nogenes is not None: results['nogenes'] = nogenes
    results['test'] = test
    results[label_col] = par

    return results


# root_dir = '/home/xoel/Escritorio/mastersthesis/'
# data_dir = root_dir + 'data/'
# scripts_dir = root_dir + 'scripts/'
# results_dir = root_dir + 'results/'
#
# genes = pd.read_csv(data_dir + 'lists/exp_aa.csv', index_col=0, header=[0, 1])
# data = pd.read_csv(data_dir + 'metaPops.tsv', sep='\t')
#
# mkt_on_df(genes.iloc[:, 50:75], data, 'aa', pops=['AFR','EUR'], tests=['aMKT', 'eMKT'], cutoffs=[0.05,0.15],
#           do_trims=[True, False])
