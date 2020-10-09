#!/bin/python3

import pandas as pd
import numpy as np
import multiprocessing as mp
import numba
from numba import jitclass, types
import mktest

class MKT:
    """
    Class implementation of a data holder and manager for McDonald-Kreitman tests.
    """
    columns = ['alpha', 'omega', 'Ka', 'Ks']
    dtypes = 'float'
    populations_dft = ['AFR', 'EUR', 'EAS', 'SAS']
    tests_dft = ['eMKT', 'aMKT']
    thresholds_dft = [[0.05, 0.15], [0, 0.1]]

    
    def __init__(self, genes, poldiv):
        self.genesets = self._get_genes(genes)
        self.poldiv = self._get_poldiv(poldiv)
        self.results = pd.DataFrame(index=[], columns=self.columns, dtype=self.dtypes)
        self.result_list = []


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
    
    
    def _get_poldiv(self, poldiv):
        new = poldiv.copy()
        new['daf0f'] = new['daf0f'].apply(self._daf_divider)
        new['daf4f'] = new['daf4f'].apply(self._daf_divider)
        return new
    

    def _daf_divider(self, dafxf):
        return list(map(int, dafxf.split(';')))
    
    def test(self, genesets=None, data=None, tests=None, thresholds=None, populations=None, label=None):
        genesets = self.genesets if not genesets else genesets
        data = self.poldiv if not data else data
        tests = self.tests_dft if not tests else tests
        thresholds = self.thresholds_dft if not thresholds else thresholds
        populations = self.populations_dft if not populations else populations
    
        self.last_result = mktest.mktest(genesets, data, tests, thresholds, populations)

        if label:
            self.last_result['label'] = label
        
        self._update_results(self.last_result)

        return self.last_result

    
    def _update_results(self, new):

        self.results = pd.concat([self.results, new])
        self.result_list.append(new)


    def amkt(self, thresholds=None, populations=None, label=None):

        return self.test(tests='aMKT', thresholds=thresholds, populations=populations, label=label)


    def emkt(self, thresholds=None, populations=None, label=None):

        return self.test(tests='eMKT', thresholds=thresholds, populations=populations, label=label)


    def bootstrap(self, n=599, tests=None, thresholds=None, populations=None, label=None):
        
        self.bs_genes = np.array([self._aux_bs(geneset, n=n) for geneset in self.genesets])
        self.bs_genes = self.bs_genes.flatten()

        return self.test(genesets=self.bs_genes, tests=tests, thresholds=thresholds, populations=populations, label=label)
        

    def _aux_bs(self, geneset, n=599):

        bs_geneset = [Geneset(geneset.name,
                      np.random.choice(geneset.geneset, size=len(geneset.geneset), replace=True)) for _ in range(n)]
    
        return bs_geneset

            
@jitclass([('name', types.string),
           ('geneset', numba.types.Array(types.UnicodeCharSeq(15), 1, 'C'))])
class Geneset:
    def __init__(self, name, geneset):
        self.name = name
        self.geneset = geneset
