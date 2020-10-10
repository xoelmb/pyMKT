# @jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def old_eMKT(daf, div, cutoff=0.15):
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




    def emkt(daf, div, cutoff):
        return