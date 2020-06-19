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


def aMKT(daf, div, xlow=0, xhigh=1, check='raise'):
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

    model = None

    try:
        ## Run asymptotic MKT and retrieve alphas
        model = amkt_fit(daf, div, xlow, xhigh, check=check)
        ratio = model['ciHigh'] / model['alpha']
        greater = model['ciHigh'] > model['alpha']
        interval = (1 <= abs(ratio) <= 10)
        if not greater or not interval:
            # print(model['alpha'], model['ciHigh'])
            # print("daf20 confidence is too wide")
            raise RuntimeError("daf20 confidence is too wide")
        res.update(model)
        res['daf10'] = False
        # print('LISTO')

    except:
        # print('trying daf10')
        daf10 = daf.copy(deep=True)
        daf10['daf'] = np.array([[x / 100, x / 100] for x in range(5, 100, 10)]).flatten()
        daf10 = daf10.groupby('daf', as_index=False).sum()
        # print(daf10)
        try:
            # print('fitting...')
            model10 = amkt_fit(daf10, div, xlow, xhigh, check=check)

            # print('did fit daf10')
            # print(model10['alpha'], model10['ciHigh'])

            # COMMENT LINES BELOW TO ENABLE COMPARISON OF DAF20 & DAF10 MODELS
            res.update(model10)
            res['daf10'] = True

            # COMMENT LINES BELOW (if-else) TO DISABLE COMPARISON
            # if model is None:
            #     print('no daf20 model')
            #     res.update(model10)
            #     res['daf10'] = True
            #
            # else:
            #     # print('comparing models')
            #     ratio10 = model['ciHigh'] / model['alpha']
            #     greater10 = model10['ciHigh'] > model10['alpha']
            #     interval10 = (1 <= abs(ratio10) <= 10)
            #     # print(f'G20: {greater};\tI20: {interval};\tG10: {greater10};\tI10: {interval10};\t')
            #     # All cases where 'failed' daf20 is better
            #     if (greater and not greater10) or (not greater and not greater10 and interval and not interval10):
            #         # print('first condition met: daf20 best')
            #         raise RuntimeError('daf20 was better')
            #
            #     # All cases where daf10 is better
            #     elif (not greater and greater10) or (greater and greater10 and not interval and interval10) or (
            #             not greater and not greater10 and not interval and interval10):
            #         # print('second condition met: daf10 best')
            #         res.update(model10)
            #         res['daf10'] = True
            #
            #     # Tricky cases: both daf10 and 20 do not comply with the two conditions (ratio in interval, ciH>alpha)
            #     elif greater and greater10:  # But none of them in interval
            #         # print('third condition met: tricky: not in interval')
            #         if model['ciHigh'] < model10['ciHigh']:
            #             raise RuntimeError('daf20 was better')
            #         else:
            #             res.update(model10)
            #             res['daf10'] = True
            #
            #     elif interval and interval10:  # But none of them ciH>alpha
            #         # print('fourth condition met: tricky: not greater')
            #         if ratio > ratio10:
            #             res.update(model10)
            #             res['daf10'] = True
            #         else:
            #             raise RuntimeError('daf20 was better')
            #     else:
            #         res.update(model10)
            #         res['daf10'] = True

        except:
            if model is not None:
                res.update(model)
                res['daf10'] = False
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


def amkt_fit(daf, div, xlow, xhigh, check='raise'):
    res = {}

    d_ratio = float(div['D0'] / div['Di'])
    # print(d_ratio)
    # Compute alpha values and trim
    alpha = 1 - d_ratio * (daf['Pi'] / daf['P0'])
    # print(alpha)
    trim = ((daf['daf'] >= xlow) & (daf['daf'] <= xhigh))

    # Two-step model fit:
    # First bounded fit:
    try:
        popt, pcov = optimize.curve_fit(exp_model, daf['daf'][trim].to_numpy(), alpha[trim].to_numpy(),
                                        bounds=([-1, -1, 1], [1, 1, 10]))
        # print('fit initial')
    except:
        # print('could not fit initial')
        popt = None
        pcov = None

    # Second fit using initially guessed values or unbounded fit:
    try:
        popt, pcov = optimize.curve_fit(exp_model, daf['daf'][trim].to_numpy(), alpha[trim].to_numpy(), p0=popt, method='lm')
        # print('Fit: lm')
    except:
        try:
            popt, pcov = optimize.curve_fit(exp_model, daf['daf'][trim].to_numpy(), alpha[trim].to_numpy(), p0=popt, method='trf')
            # print('Fit: trf')
        except:
            try:
                popt, pcov = optimize.curve_fit(exp_model, daf['daf'][trim].to_numpy(), alpha[trim].to_numpy(), p0=popt,
                                           method='dogbox')
                # print('Fit: dogbox')
            except:
                if not popt:
                    # print('Could not fit any unbounded')
                    raise RuntimeError("Couldn't fit any method")

    res['a'] = popt[0]
    res['b'] = popt[1]
    res['c'] = popt[2]

    # alpha for predicted model
    res['alpha'] = exp_model(1.0, res['a'], res['b'], res['c'])

    # Compute confidence intervals based on simulated data (MC-SOERP)
    vcov = pd.concat([pd.DataFrame([0] * 4).transpose(),
                      pd.concat([pd.DataFrame([0] * 4), pd.DataFrame(pcov)], axis=1, ignore_index=True)],
                     axis=0, ignore_index=True)
    vcov = vcov.iloc[0:4, :].values

    simpars = np.random.multivariate_normal(mean=[1.0, res['a'], res['b'], res['c']], cov=vcov, size=10000,
                                            check_valid=check)  # check_valid=raise -> same as R implementation

    res['ciLow'], res['ciHigh'] = np.quantile([exp_model(x[0], x[1], x[2], x[3]) for x in simpars], [0.025, 0.975])

    return res


@jit(nopython=True)  # Set "nopython" mode for best performance, equivalent to @njit
def exp_model(f_trimmed, a, b, c):
    return a + b * np.exp(-c * f_trimmed)


def mkt_on_df(gene_df, data_df, label=None, pops=None, tests=None, cutoffs=None, do_trims=None, bootstrap=None,
              b_size=100, b_reps=100):
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

    pars = [(gene_df.iloc[:, i], data_df, pops, tests, cutoffs, do_trims, bootstrap, b_size, b_reps) for i in
            range(len(gene_df.columns.values))]

    # Loads the models for all the parameters parsed using multiprocessing to speed up computations
    pool = MyPool(processes=8)  # multiprocessing.cpu_count())
    results_list = pool.starmap(mkt_on_col, pars)
    pool.terminate()
    results = pd.concat(results_list, axis=0, ignore_index=True)

    if label is not None: results['label'] = label

    return results


def mkt_on_col(col, data_df, pops=None, tests=None, cutoffs=None, do_trims=None, bootstrap=None, b_size=100,
               b_reps=100):
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

    glist = col[col == 1].index.values
    if len(glist) > 0:
        if bootstrap:
            results = bootstrap_on_list(glist, data_df, pops, tests, cutoffs, do_trims, b_size, b_reps)
        else:
            results = mkt_on_list(glist, data_df, pops, tests, cutoffs, do_trims)
    else:
        results = pd.DataFrame(index=[0])
        results['nogenes'] = 0

    if col.name is not None:
        results['stage'] = col.name[0]
        results['region'] = col.name[1]
    return results


def bootstrap_on_list(glist, data_df, pops=None, tests=None, cutoffs=None, do_trims=None, b_size=100, b_reps=100):
    pars = [(np.random.choice(glist, b_size), data_df, pops, tests, cutoffs, do_trims) for _ in range(b_reps)]

    pool = MyPool(processes=4)  # multiprocessing.cpu_count())
    results_list = copy.deepcopy(pool.starmap(mkt_on_daf, pars))
    pool.terminate()

    results = pd.concat(results_list, axis=0, ignore_index=True)
    return results


def mkt_on_list(glist, data_df, pops=None, tests=None, cutoffs=None, do_trims=None):
    if do_trims is None:
        do_trims = [True, False]
    if cutoffs is None:
        cutoffs = [0.05, 0.15]
    if tests is None:
        tests = ['eMKT', 'aMKT']
    if pops is None:
        pops = ['AFR', 'EUR']

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
            results = copy.deepcopy(aMKT(daf, div, xlow, xhigh, check='raise'))
            if 'alpha' not in results.keys():  # Uses covar matrix even if it's not valid, leads to bad CI's
                results = copy.deepcopy(aMKT(daf, div, xlow, xhigh, check='ignore'))

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

### DEBUGGING ###
#
# root_dir = '/home/xoel/Escritorio/mastersthesis/'
# data_dir = root_dir + 'data/'
# scripts_dir = root_dir + 'scripts/'
# results_dir = root_dir + 'results/'
# #
# genes = pd.read_csv(data_dir + 'lists/exp_aa.csv', index_col=0, header=[0, 1])
# data = pd.read_csv(data_dir + 'metaPops.tsv', sep='\t')
#
#
#
# debug = mkt_on_df(genes.iloc[:, 0:16], data, 'aa', pops=['AFR', 'EUR'], tests=['aMKT', 'eMKT'], cutoffs=[0.05, 0.15],
#           do_trims=[True, False], bootstrap=False, b_size=100, b_reps=100)
