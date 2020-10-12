import pandas as pd
import numpy as np
import pathos.multiprocessing as mp
import sfs
import emkt
import amkt

n_jobs = mp.cpu_count()+2

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



     



