from numba import jit, types, vectorize
import numpy as np
from pvalue import pval
import fisher


@jit(nopython=False) # Set "nopython" mode for best performance, equivalent to @njit
def old_emkt(daf, div, cutoff=0.15):
    res = {}

    P0 = daf['P0'].sum()
    Pi = daf['Pi'].sum()
    D0 = div['D0']
    Di = div['Di']
    m0 = div['m0']
    mi = div['mi']

    # Divergence metrics
    res['Ka'] = Di / mi
    res['Ks'] = D0 / m0
    res['omega'] = res['Ka'] / res['Ks']

    ### Estimating alpha with Pi/P0 ratio
    PiMinus = daf['Pi'][daf['daf'] <= cutoff].sum()
    PiGreater = daf['Pi'][daf['daf'] > cutoff].sum()
    P0Minus = daf['P0'][daf['daf'] <= cutoff].sum()
    P0Greater = daf['P0'][daf['daf'] > cutoff].sum()

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

    # res['pvalue'] = fisher.pvalue(P0, D0, Pi - deleterious, Di).two_tail
    res['pvalue'] = pval(P0, D0, Pi - deleterious, Di)

    ## Omega A and Omega D
    res['omegaA'] = res['omega'] * res['alpha']
    res['omegaD'] = res['omega'] - res['omegaA']

    return res


@jit(nopython=True)
def v_emkt(P0, Pi, D0, Di, m0, mi, threshold=0.15, f=np.arange(0.025,0.985,0.05)):

    idx = int((threshold-0.025)/0.05)
    tP0 = P0.sum()
    tPi = Pi.sum()
    
    
    # Divergence metrics
    Ka = Di / mi
    Ks = D0 / m0
    omega = Ka / Ks
     

    ### Estimating alpha with Pi/P0 ratio
    PiMinus = Pi[f <= threshold].sum()
    PiGreater = Pi[f > threshold].sum()
    P0Minus = P0[f <= threshold].sum()
    P0Greater = P0[f > threshold].sum()


    ratioP0 = P0Minus / P0Greater
    deleterious = PiMinus - (PiGreater * ratioP0)
    PiNeutral = tPi - deleterious

    alpha = 1 - (((tPi - deleterious) / tP0) * (D0 / Di))

    ## Estimation of b: weakly deleterious
    neg_b = (deleterious / tP0) * (m0 / mi)

    ## Estimation of f: neutral sites
    neg_f = (m0 * PiNeutral) / (mi * tP0)

    ## Estimation of d, strongly deleterious sites
    neg_d = 1 - (neg_f + neg_b)

    pvalue = pval(tP0, D0, tPi - deleterious, Di)

    ## Omega A and Omega D
    omegaA = omega * alpha
    omegaD = omega - omegaA


    return (('Ka', Ka),
            ('Ks', Ks),
            ('alpha', alpha),
            ('omega', omega),
            ('omegaA', omegaA),
            ('omegaD', omegaD),
            ('neg_b', neg_b),
            ('neg_d', neg_d),
            ('neg_f', neg_f),
            ('pvalue', pvalue))


@jit(nopython=False)
def emkt(P0, Pi, D0, Di, m0, mi, threshold=0.15, f=np.arange(0.025,0.985,0.05)):

    idx = int((threshold-0.025)/0.05)
    tP0 = P0.sum()
    tPi = Pi.sum()
    
    # f = np.arange(0.025,0.985,0.05)
    
    # Divergence metrics
    Ka = Di / mi
    Ks = D0 / m0
    omega = Ka / Ks
     

    ### Estimating alpha with Pi/P0 ratio
    PiMinus = Pi[f <= threshold].sum()
    PiGreater = Pi[f > threshold].sum()
    P0Minus = P0[f <= threshold].sum()
    P0Greater = P0[f > threshold].sum()


    ratioP0 = P0Minus / P0Greater
    deleterious = PiMinus - (PiGreater * ratioP0)
    PiNeutral = tPi - deleterious

    alpha = 1 - (((tPi - deleterious) / tP0) * (D0 / Di))

    ## Estimation of b: weakly deleterious
    neg_b = (deleterious / tP0) * (m0 / mi)

    ## Estimation of f: neutral sites
    neg_f = (m0 * PiNeutral) / (mi * tP0)

    ## Estimation of d, strongly deleterious sites
    neg_d = 1 - (neg_f + neg_b)

    pvalue = pval(tP0, D0, tPi - deleterious, Di)

    ## Omega A and Omega D
    omegaA = omega * alpha
    omegaD = omega - omegaA


    return (('Ka', Ka),
            ('Ks', Ks),
            ('alpha', alpha),
            ('omega', omega),
            ('omegaA', omegaA),
            ('omegaD', omegaD),
            ('neg_b', neg_b),
            ('neg_d', neg_d),
            ('neg_f', neg_f),
            ('pvalue', pvalue))




@jit(nopython=True)
def d_emkt(P0, Pi, D0, Di, m0, mi, threshold=0.15, f=np.arange(0.025,0.985,0.05)):
    res = {}

    tP0 = P0.sum()
    tPi = Pi.sum()
    
    
    # Divergence metrics
    res['Ka'] = Di / mi
    res['Ks'] = D0 / m0
    res['omega'] = res['Ka'] / res['Ks']
     


    ### Estimating alpha with Pi/P0 ratio
    PiMinus = Pi[f <= threshold].sum()
    PiGreater = Pi[f > threshold].sum()
    P0Minus = P0[f <= threshold].sum()
    P0Greater = P0[f > threshold].sum()


    ratioP0 = P0Minus / P0Greater
    deleterious = PiMinus - (PiGreater * ratioP0)
    PiNeutral = tPi - deleterious

    res['alpha'] = 1 - (((tPi - deleterious) / tP0) * (D0 / Di))

    ## Estimation of b: weakly deleterious
    res['neg_b'] = (deleterious / tP0) * (m0 / mi)

    ## Estimation of f: neutral sites
    res['neg_f'] = (m0 * PiNeutral) / (mi * tP0)

    ## Estimation of d, strongly deleterious sites
    res['neg_d'] = 1 - (res['neg_f'] + res['neg_b'])

    res['pvalue'] = pval(tP0, D0, tPi - deleterious, Di)

    ## Omega A and Omega D
    res['omegaA'] = res['omega'] * res['alpha']
    res['omegaD'] = res['omega'] - res['omegaA']


    return res

