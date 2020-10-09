import pandas as pd
import numpy as np
import multiprocessing as mp
import sfs

n_jobs = mp.cpu_count()

def mktest(genesets, data, tests, thresholds, populations):
     return sfs.parallel_sfs(genesets, data, tests, populations)
     



