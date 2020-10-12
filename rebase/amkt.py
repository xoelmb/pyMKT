from numba import jit
from numba.types import float64
import numpy as np
from scipy import optimize


@jit(nopython=True)  # Set "nopython" mode for best performance, equivalent to @njit
def exp_model(f_trimmed, a, b, c):
    return a + b * np.exp(-c * f_trimmed)


def amkt(daf, div, xlow=0, xhigh=1, f=np.arange(0.025,0.985,0.05), check='raise'):
    res = {}

    P0 = daf['P0'].sum()
    Pi = daf['Pi'].sum()
    D0 = div['D0']
    Di = div['Di']
    m0 = div['m0']
    mi = div['mi']

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
        model = amkt_fit(daf, div, xlow, xhigh, f, check=check)
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
        daf10 = {'P0': daf['P0'].reshape((10,2)).sum(axis=1),
                 'Pi': daf['Pi'].reshape((10,2)).sum(axis=1)}
        # print(daf10)
        try:
            # print('fitting...')
            model10 = amkt_fit(daf10, div, xlow, xhigh, f, check=check)

            # print('did fit daf10')
            # print(model10['alpha'], model10['ciHigh'])

            # COMMENT LINES BELOW TO ENABLE COMPARISON OF DAF20 & DAF10 MODELS
            res.update(model10)
            res['daf10'] = True

        except:
            if model is not None:
                res.update(model)
                res['daf10'] = False
            else:
                return res

    # Estimate the fraction of slightly deleterious sites in each daf category (b)
    omegaD = daf['Pi'] - (((1 - res['alpha']) * Di * daf['P0']) / D0)
    res['neg_b'] = (omegaD.sum() / daf['P0'].sum()) * (m0 / mi)

    # Re-estimate the truly number of neutral sites, removing the slightly deleterious
    res['neg_f'] = fb - res['neg_b']

    ## Omega A and Omega
    res['omegaA'] = res['omega'] * res['alpha']
    res['omegaD'] = res['omega'] - res['omegaA']

    return res


def amkt_fit(daf, div, xlow=0, xhigh=1, f=np.arange(0.025,0.985,0.05), check='raise'):
    res = {}

    d_ratio = float(div['D0']/div['Di'])
    # print(d_ratio)
    # Compute alpha values and trim
    alpha = 1 - d_ratio * (daf['Pi']/daf['P0'])
    # print(alpha)
    trim = ((f >= xlow) & (f <= xhigh))

    # Two-step model fit:
    # First bounded fit:
    try:
        popt, pcov = optimize.curve_fit(exp_model, f[trim], alpha[trim],
                                        bounds=([-1, -1, 1], [1, 1, 10]))
        # print('fit initial')
    except:
        # print('could not fit initial')
        popt = None
        pcov = None

    # Second fit using initially guessed values or unbounded fit:
    for method in ['lm', 'trf', 'dogbox']:
        try:
            popt, pcov = optimize.curve_fit(exp_model, f[trim], alpha[trim],
                                            p0=popt, method=method)
            res['method'] = method
            break
        except:
            continue
        
    if popt is None:
        # print('Could not fit any unbounded')
        raise RuntimeError("Couldn't fit any method")

    res['a'] = popt[0]
    res['b'] = popt[1]
    res['c'] = popt[2]

    # alpha for predicted model
    res['alpha'] = exp_model(1.0, res['a'], res['b'], res['c'])

    # Compute confidence intervals based on simulated data (MC-SOERP)
    vcov = np.append([[0,0,0,0]], np.append([[0],[0],[0]], pcov, 1), 0)

    simpars = np.random.multivariate_normal(mean=[1.0, res['a'], res['b'], res['c']], cov=vcov, size=10000,
                                            check_valid=check)  # check_valid=raise -> same as R implementation

    res['ciLow'], res['ciHigh'] = np.quantile([exp_model(x[0], x[1], x[2], x[3]) for x in simpars], [0.025, 0.975])

    return res