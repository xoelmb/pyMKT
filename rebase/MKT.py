#!/bin/python3

import pandas as pd
import numpy as np
import mktest
import multiprocessing as mp
import psutil



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

    
    def __init__(self, genes, poldiv):
        self.mean_genes = genes.sum(axis=0).mean()
        self.nsamples = len(genes.columns.values)
        self.bs_factor = self.mean_genes*self.nsamples
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

        col_types = dict(list(map(lambda x: (x, int),['mi', 'm0', 'pi', 'p0', 'di', 'd0'])))

        new = poldiv.astype(col_types)
        
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
        
        self.last_result = pd.DataFrame(mktest.mktest(genesets, data, tests, thresholds, populations))

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


    def bootstrap(self, n=599, tests=None, thresholds=None, populations=None, label=None, max_ram=5e9):

        def compute_lim(max_ram=max_ram, factor=None, intercept=1.51294, slope=1.08):
            
            if not factor: factor = self.bs_factor
            return int(10**((np.log10(max_ram)-intercept)/slope - np.log10(factor)))


        def aux(geneset, n=599):
            bs_geneset = np.random.choice(geneset[1], size=(n, len(geneset[1])), replace=True)
            return [(geneset[0], x) for x in bs_geneset]


        results = pd.DataFrame()
        lim = compute_lim(max_ram)
        # print(lim)
                
        if n >= lim:
            print(f'Memory requirements exceed max_ram ({round(max_ram/10**9, 2)} GB).\nSplitting {n} repetitions in sets of {lim}.\n')
            for i in range(n//lim):
                print(f'Running {i*lim}-{(i+1)*lim}')
                bs_genesets = [aux(geneset, lim) for geneset in self.genesets]
                bs_genesets = [item for sublist in bs_genesets for item in sublist]
                last = self.test(genesets=bs_genesets, tests=tests, thresholds=thresholds, populations=populations, label=label)
                results = pd.concat([results, last], axis=0, ignore_index=True)
            print(f'Running {(n//lim)*lim}-{n}')

        bs_genesets = [aux(geneset, n%lim) for geneset in self.genesets]
        # bs_genesets = [aux(geneset, n) for geneset in self.genesets]
        bs_genesets = [item for sublist in bs_genesets for item in sublist]
        last = self.test(genesets=bs_genesets, tests=tests, thresholds=thresholds, populations=populations, label=label)
        self.last_result = pd.concat([results, last], axis=0, ignore_index=True)
        
        return self.last_result



