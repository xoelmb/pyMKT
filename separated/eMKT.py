#!/usr/bin/env python

from numba import jit
from fisher import pvalue as pval
from collections import namedtuple

def eMKT(daf, div, cutoff):
    res = {}
    res['alpha'], res['Ka'], res['Ks'], res['omega'], res['omegaA'], res['omegaD'], res['neg_b'], res['neg_d'], \
    res['neg_f'], contingency = fast_eMKT(namedtuple('daf', daf.keys())(*daf.values()), namedtuple('div', div.keys())(*div.values()), cutoff)

    res['pvalue'] = pval(*contingency).two_tail

    return res


@jit(nopython=True)
def fast_eMKT(daf, div, cutoff):
    Pi = daf.pi.sum()
    P0 = daf.p0.sum()

    Ka = div.Di / div.mi
    Ks = div.D0 / div.m0
    omega = Ka / Ks

    # Estimating alpha with Pi/P0 ratio
    PiMinus = daf.pi[daf.f <= cutoff].sum()
    PiGreater = daf.pi[daf.f > cutoff].sum()
    P0Minus = daf.p0[daf.f <= cutoff].sum()
    P0Greater = daf.p0[daf.f > cutoff].sum()

    ratioP0 = P0Minus / P0Greater
    deleterious = PiMinus - (PiGreater * ratioP0)
    PiNeutral = Pi - deleterious

    alpha = 1 - (((Pi - deleterious) / P0) * (div.D0 / div.Di))

    # Estimation of b: weakly deleterious
    neg_b = (deleterious / P0) * (div.m0 / div.mi)

    # Estimation of f: neutral sites
    neg_f = (div.m0 * PiNeutral) / (div.mi * P0)

    # Estimation of d, strongly deleterious sites
    neg_d = 1 - (neg_f + neg_b)

    # Omega A and Omega D
    omegaA = omega * alpha
    omegaD = omega - omegaA

    # Save contingency table to compute pvalue out of nopython mode
    contingency = P0, div.D0, Pi - deleterious, div.Di

    return alpha, Ka, Ks, omega, omegaA, omegaD, neg_b, neg_d, neg_f, contingency
