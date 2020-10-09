from numba import float64, int64, jitclass
import numpy as np

@jitclass([('daf', float64[::1]),
           ('Pi', int64[:]),
           ('P0', int64[:])])
class DAF:
    def __init__(self, Pi, P0, daf=np.arange(0.025, 0.980, 0.05)):
        self.daf = daf
        self.Pi = Pi
        self.P0 = P0


@jitclass([('mi', int64),
           ('m0', int64),
           ('Di', int64),
           ('D0', int64)])
class DIV:
    def __init__(self, mi, m0, Di, D0):
        self.Di = Di
        self.D0 = D0
        self.mi = mi
        self.m0 = m0
