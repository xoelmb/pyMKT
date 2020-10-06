#!/bin/python3

import pandas as pd
import numpy as np
import multiprocessing as mp

class MKT:
    """
    Class implementation of a data holder and manager for McDonald-Kreitman tests.
    """
    columns = ['alpha', 'omega', 'Ka', 'Ks']
    dtypes = 'float'
    populations = ['AFR', 'EUR', 'EAS', 'SAS']
    tests = ['eMKT', 'aMKT']
    thresholds = [[0.05, 0.15], [0, 0.1]]

    
    def __init__(self, genes, poldiv):
        self.g = self._get_genes(genes)
        self.ph = poldiv
        self.results = pd.DataFrame(index=[], columns=self.columns, dtype=self.dtypes)
        self.result_list = []


    def _get_genes(self, genes):
        """
        Adapt input data. Accepted input types: pandas.DataFrame(), numpy.array().
        """            
        if isinstance(genes, pd.DataFrame):
            return genes.apply(self._col_to_array, axis=0).T.values

        if isinstance(genes, np.array):
            return np.array(genes, str)
        
        else:
            raise TypeError('df or np.array(strings)')


    def _col_to_array(self, col):
        return np.array(('_'.join(col.name), np.array(col.index[col == 1].values, str)))

    
    def test(self, genes=self.g, data=self.ph, tests=self.tests, thresholds=self.thresholds, populations=self.populations, label=None):
        
        self.last_result = mktest(genes, data, tests, thresholds, populations)

        if label:
            self.last_result['label'] = label
        
        self.update_results(self.last_result)

        return self.last_result

    
    def _update_results(self, new):

        self.results = pd.concat([self.results, new])
        self.result_list.append(new)


    def amkt(self, genes=self.g, data=self.ph, thresholds=self.thresholds, populations=self.populations, label=None):

        return test(genes=genes, data=data, tests='aMKT', thresholds=thresholds, populations=populations, label=label)


    def emkt(self, genes=self.g, data=self.ph, thresholds=self.thresholds, populations=self.populations, label=None):

        return test(genes=genes, data=data, tests='eMKT', thresholds=thresholds, populations=populations, label=label)


    def bootstrap(self, n=599, genes=self.g, data=self.ph, tests=self.tests, thresholds=self.thresholds, populations=self.populations, label=None):

        bs_genes = np.array([aux_bs(geneset, n=n) for geneset in genes])
        
        return test(genes=bs_genes, data=data, tests=['eMKT', 'aMKT'], thresholds=thresholds, populations=populations, label=label)
        

    def aux_bs(geneset, n=599):

        bs_geneset = [np.array((geneset[0], np.random.choice(geneset[1],
                    size=len(geneset[1]), replace=True))) for _ in range(n)]
    
        return np.array(bs_geneset)

            
