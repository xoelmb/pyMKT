import pandas as pd
import numpy as np
import pathos.multiprocessing as mp
import sfs
import emkt
import amkt

n_jobs = mp.cpu_count()+mp.cpu_count()//2

def mktest(genesets, popdata, tests, thresholds, populations):
     print("· [1/2] Computing polymorphism & divergence ", end='')
     poldivs = sfs.parallel_sfs(genesets, popdata, tests, thresholds, populations)
     print('[DONE]')
     par_expander = lambda x: mkt_caller(**x)


     print("· [2/2] Running tests ", end='')
     mypool = mp.Pool(n_jobs)
     results = mypool.map(par_expander, poldivs)
     mypool.terminate()
     print('[DONE]')


     return results


def mkt_caller(daf, div, test, threshold, name=None, population=None):

     results = {}

     if test == 'eMKT':
          results = emkt.emkt(daf, div, cutoff=threshold)

     elif test == 'aMKT':
          results = amkt.amkt(daf, div, xlow=threshold, xhigh=1-threshold)
          if 'alpha' not in results.keys():  # Uses covar matrix even if it's not valid, leads to bad CI's
               results = amkt.amkt(daf, div, xlow=threshold, xhigh=1-threshold, check='ignore')

     else:
          raise RuntimeError('test not available')

     for k, v in zip(['name', 'population', 'test', 'threshold'], [name, population, test, threshold]):
          if v:
               results[k] = v

     return results



     



