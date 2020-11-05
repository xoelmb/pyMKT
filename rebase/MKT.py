#!/bin/python3

import pandas as pd
import numpy as np
import mktest
import multiprocessing as mp



class MKT:
    """
    Class implementation of a data holder and manager for McDonald-Kreitman tests.
    """
    columns = ['alpha', 'omega', 'Ka', 'Ks']
    dtypes = 'float'
    populations_dft = ['AFR', 'EUR', 'EAS', 'SAS']
    tests_dft = ['eMKT', 'aMKT']
    thresholds_dft = [[0.05, 0.15], [0, 0.1]]
    bootstrap_lim_dft = 50

    
    def __init__(self, genes, popdata, frac=False):

        self.genes = genes.sample(frac=frac, axis=1) if frac else genes

        self.mean_genes = genes.sum(axis=0).mean()
        self.nsamples = len(genes.columns.values)
        self.bs_factor = self.mean_genes*self.nsamples

        self.genesets = self._get_genes(genes)
        self.popdata = self._get_popdata(popdata)

        self.results = pd.DataFrame(index=[], columns=self.columns, dtype=self.dtypes)
        self.result_list = []


    def _set_debug(self, mode=0.1):
        self.populations_dft = ['AFR']
        self.tests_dft = ['eMKT']
        self.thresholds_dft = [[0.15]]
        self.genes = self.genes.sample(frac=mode, axis=1)



    def _get_genes(self, genes):
        """
        Adapt input data. Accepted input types: pandas.DataFrame(), numpy.array().
        """            
        if isinstance(genes, pd.DataFrame):
            return genes.apply(self._col_to_array, axis=0).T.values

        if isinstance(genes, np.array):
            return ('unnamed', np.array(genes, str))
        
        else:
            raise TypeError('df or np.array(strings)')


    def _col_to_array(self, col):
        return (col.name, np.array(col.index[col == 1].values, str))
    
    
    def _get_popdata(self, popdata):

        col_types = dict(list(map(lambda x: (x, int),['mi', 'm0', 'pi', 'p0', 'di', 'd0'])))

        new = popdata.astype(col_types)

        new[[f'Pi{i}' for i in range(20)]] = new['daf0f'].str.split(';',expand=True).astype(int)
        new[[f'P0{i}' for i in range(20)]] = new['daf4f'].str.split(';',expand=True).astype(int)
        
        return dict(list(new.groupby('pop')))
    

    def test(self, genesets=None, popdata=None, tests=None, thresholds=None, populations=None,
             label=None, reps=100, permute=False, bootstrap=False,
             permute_vars_alone=False, permute_vars_and_constant=True,
             variable_genes=None, v=True, c=25):

        genesets = self.genesets if not genesets else genesets
        popdata = self.popdata if not popdata else popdata
        tests = self.tests_dft if not tests else tests
        thresholds = self.thresholds_dft if not thresholds else thresholds
        populations = self.populations_dft if not populations else populations

        red_popdata = {pop: popdata[pop] for pop in populations}
        
        if permute:
            genesets = np.array([np.array([g[0],g[1], variable_genes]) for g in genesets])
            
        self.last_result = pd.DataFrame(mktest.mktest(genesets,
                                                      red_popdata,
                                                      tests,
                                                      thresholds,
                                                      permute=permute,
                                                      bootstrap=bootstrap,
                                                      reps=reps,
                                                      v=v,
                                                      permute_vars_alone=permute_vars_alone,
                                                      permute_vars_and_constant=permute_vars_and_constant,
                                                      c=c))

        if label:
            self.last_result['label'] = label
        
        self._update_results(self.last_result)

        return self.last_result

    
    def _update_results(self, new):

        self.results = pd.concat([self.results, new], axis=0, ignore_index=True)
        self.result_list.append(new)


    def amkt(self, thresholds=None, populations=None, label=None):

        return self.test(tests='aMKT', thresholds=thresholds, populations=populations, label=label)


    def emkt(self, thresholds=None, populations=None, label=None):

        return self.test(tests='eMKT', thresholds=thresholds, populations=populations, label=label)

