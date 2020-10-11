import pandas as pd
import numpy as np
import pathos.multiprocessing as mp
import sfs
import tests
import time

n_jobs = mp.cpu_count()

def mktest(genesets, data, tests, thresholds, populations):
     poldivs = sfs.parallel_sfs(genesets, data, tests, populations)
     
     pars = par_builder(poldivs, tests, thresholds)
     par_expander = lambda x: mkt_caller(**x)

     mypool = mp.Pool(n_jobs+2)
     results = mypool.map(par_expander, pars)
     mypool.terminate()

     return results


def par_builder(poldivs, tests, thresholds):
     pars = []
     for t, ths in zip(tests, thresholds):
          use_daf = 'daf_cum' if t == 'aMKT' else 'daf'
          for th in ths:
               for pd in poldivs:
                    par = dict(name=pd[0],
                               population=pd[1],
                               daf=pd[2][use_daf],
                               div=pd[2]['div'],
                               test=t,
                               threshold=th)
                    pars.append(par)
     return pars


def mkt_caller(daf, div, test, threshold, name=None, population=None):
     test='nold_eMKT'
     f = np.arange(0.025,0.985, 0.05)
     if test == 'old_eMKT':
          daf['daf'] = f
          t0=time.time()
          results = tests.old_emkt(daf, div, cutoff=threshold)
          return    time.time()-t0
     elif test == 'nold_eMKT':
          t0=time.time()
          results = tests.nold_emkt(daf, div, cutoff=threshold, f=f)
          return    time.time()-t0
     elif test == 'eMKT':
          t0=time.time()
          results = dict(tests.emkt(**daf, **div, threshold=threshold, f=f))
          return time.time()-t0
     elif test == 'v_eMKT':
          t0=time.time()
          results = dict(tests.v_emkt(**daf, **div, threshold=threshold, f=f))
          return time.time()-t0
     elif test == 'd_eMKT':
          t0=time.time()
          results = tests.emkt(**daf, **div, threshold=threshold, f=f)
          return time.time()-t0


     else:
          raise RuntimeError('test not available')
     
          

     for to_name in [name, population]:
          if to_name:
               results[to_name] = to_name

     return results



     



