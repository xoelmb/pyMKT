root_dir = '/home/xoel/Escritorio/mastersthesis/'
data_dir = root_dir+'data/'
lists_dir = data_dir+'lists/'
scripts_dir = root_dir+'scripts/'
results_dir = root_dir+'results/'
plots_dir = root_dir+'plots/'

import pandas as pd
import numpy as np
import MKT
from timeit import timeit

genes = pd.read_csv(lists_dir+'exp_aa.csv', header=[0,1], index_col=0)
genes.columns = list(map(lambda x: '_'.join(x), genes.columns.values))
ph = pd.read_csv(data_dir+'metaPops.tsv', sep='\t')


a = MKT.MKT(genes, ph)

r = a.test()



# r = pd.DataFrame()

# a = MKT.MKT()

# for lim, n in zip([10, 50, 100],[35, 175, 350]):
#     for pop in [['AFR'], ['EUR', 'AFR']]:
#         for t in[['eMKT']]:
#             for th in [[[0]], [[0, 0.1]]]:

#                 MKT.MKT.bootstrap_lim_dft = lim
#                 MKT.MKT.populations_dft = pop
#                 MKT.MKT.tests_dft = t
#                 MKT.MKT.thresholds_dft = th
#                 

#                 setup='from __main__ import a'
#                 time = timeit(f'a.bootstrap({n})', setup=setup, number=1)
#                 df = pd.DataFrame([{'lim': lim,
#                                     'bs': n,
#                                     'pops': len(pop),
#                                     'tests': len(t),
#                                     'ths': len(th[0]),
#                                     'total': len(a.last_result),
#                                     'time': time}])
#                 print(df)
#                 r = pd.concat([r, df],
#                               ignore_index=True, axis=0)

# r