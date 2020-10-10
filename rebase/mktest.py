import pandas as pd
import numpy as np
import pathos.multiprocessing as mp
import sfs
import tests

n_jobs = mp.cpu_count()

def mktest(genesets, data, tests, thresholds, populations):
     poldivs = sfs.parallel_sfs(genesets, data, tests, populations)
     
     pars = par_builder(poldivs, tests, thresholds)

     par_expander = lambda x: mkt_caller(**x)

     mypool = mp.Pool(n_jobs)
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
     return (daf, div, threshold)
     # if test == 'aMKT':
     #      results = dict(tests.amkt(**daf, **div, threshold=threshold))
     # elif test == 'eMKT':
     #      results = dict(tests.emkt(**daf, **div, threshold=threshold))
     # else:
     #      raise RuntimeError('test not available')
     
          

     # for to_name in [name, population]:
     #      if to_name:
     #           results[to_name] = to_name

     # return results



     



