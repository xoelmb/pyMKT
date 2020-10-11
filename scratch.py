from numba import jit, typeof
from numba.typed import Dict
from numba.types import int64, string, UniTuple

d_type = typeof(Dict.empty(string, int64))

def gen_p(a,b):
    return {'a': a, 'b': b}


def gen_n(a,b):
    div_dict = Dict.empty(string, int64)
    div_dict['a'] = a
    div_dict['b'] = b
    return div_dict


def rec_p(d):
    return d['a'], d['b']


@jit(UniTuple(int64, 2)(d_type), nopython=True)
def rec_n(d):
    return d['a'], d['b']


@jit(UniTuple(int64, 2)(int64, int64), nopython=True)
def rec_en(a,b):
    return a,b


def rec_ep(a,b):
    return a,b


def p_p(a,b):
    d = gen_p(a,b)
    return rec_p(d)

def p_p(a,b):
    d = gen_p(a,b)
    return rec_p(d)


def n_n(a,b):
    d = gen_n(a,b)
    return rec_n(d)
    

def p_en(a,b):
    d = gen_p(a,b)
    return rec_en(**d)


def p_ep(a,b):
    d = gen_p(a,b)
    return rec_ep(**d)
    