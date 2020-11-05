import numpy as np
from pvalue import pval


def emkt(daf, div, cutoff=0.15, f=np.arange(0.025,0.985,0.05)):
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
    PiMinus = daf['Pi'][f <= cutoff].sum()
    PiGreater = daf['Pi'][f > cutoff].sum()
    P0Minus = daf['P0'][f <= cutoff].sum()
    P0Greater = daf['P0'][f > cutoff].sum()

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
    try:
        res['pvalue'] = pval(P0, D0, Pi - deleterious, Di)
    except:
        pass
    
    ## Omega A and Omega D
    res['omegaA'] = res['omega'] * res['alpha']
    res['omegaD'] = res['omega'] - res['omegaA']

    return res