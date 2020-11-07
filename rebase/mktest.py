import pandas as pd
import numpy as np
import pathos.multiprocessing as mp
import sfs
import emkt
import amkt
import sys, time

n_jobs = mp.cpu_count()#+mp.cpu_count()//2

def mktest(genesets, popdata, tests, thresholds, permute=False, bootstrap=False, reps=100, v=True, permute_vars_alone=False, permute_vars_and_constant=True):

     poldivs = sfs.parallel_sfs(genesets, popdata, tests, thresholds, permute=permute, bootstrap=bootstrap, reps=reps, v=v, permute_vars_alone=permute_vars_alone, permute_vars_and_constant=permute_vars_and_constant)

     par_expander = lambda x: mkt_caller(**x)
     mypool = mp.Pool(n_jobs)

     if v:
          n=len(poldivs)
          results = []
          sys.stderr.write('· [2/2] Running tests...')

          for i, r in enumerate(mypool.imap_unordered(par_expander, poldivs, chunksize=n//n_jobs+1), 1):
               results.append(r)
               sys.stderr.write(f'\r· [2/2] Running tests {round(i/n*100,2)}%')
          sys.stderr.write('\r· [2/2] Running tests [DONE]\n')
     else:
          results = list(mypool.imap_unordered(par_expander, poldivs, chunksize=n//n_jobs+1))

     mypool.terminate()

     return results


def mkt_caller(daf, div, test, threshold, name=None, population=None, ngenes=None, repeat=None):

     results = {}

     if test == 'eMKT':
          results = emkt.emkt(daf, div, cutoff=threshold)

     elif test == 'aMKT':
          results = amkt.amkt(daf, div, xlow=threshold, xhigh=1-threshold)
          if 'alpha' not in results.keys():  # Uses covar matrix even if it's not valid, leads to bad CI's
               results = amkt.amkt(daf, div, xlow=threshold, xhigh=1-threshold, check='ignore')

     else:
          raise RuntimeError('test not available')

     for k, v in zip(['name', 'population', 'test', 'threshold', 'ngenes', 'repeat'], [name, population, test, threshold, ngenes, repeat]):
          if v is not None:
               results[k] = v

     return results



     



