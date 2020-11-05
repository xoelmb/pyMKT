import pandas as pd
import numpy as np
import MKT
import timeit, time
from permutator import permutator_mkt

root_dir = '/home/xoel/Escritorio/mastersthesis/'
data_dir = root_dir+'data/'
lists_dir = data_dir+'lists/'
scripts_dir = root_dir+'scripts/'
results_dir = root_dir+'results/'
plots_dir = root_dir+'plots/'

genes = pd.read_csv(lists_dir+'exp_aa.csv', header=[0,1], index_col=0)
genes.columns = list(map(lambda x: '_'.join(x), genes.columns.values))
ph = pd.read_csv(data_dir+'metaPops.tsv', sep='\t')

# a = MKT.MKT(genes, ph)

# t0=time.time()
# r = a.test()
# time.sleep(1)
# print(time.time()-t0)

# t0=time.time()
# r = a.test(bootstrap=True, reps=10)
# time.sleep(1)
# print(time.time()-t0)

#################################################################################


#################################################################################
########### CHECK IF RESULTS ARE WHAT THEY SHOULD FROM VALIDATED DATA ###########
#################################################################################

#################################################################################
pops=['EUR', 'AFR', 'EAS', 'SAS']
tests=['eMKT', 'aMKT']
thresholds=[[0.05, 0.15], [0, 0.1]]

MKT.MKT.populations_dft = pops
MKT.MKT.tests_dft = tests
MKT.MKT.thresholds_dft = thresholds
#################################################################################

a = MKT.MKT(genes, ph)
r = a.test()


val_file = '/home/xoel/Escritorio/pyMKT/val_results.csv'
val_results = pd.read_csv(val_file)
val_results['name'] = val_results[['stage','region']].apply(lambda x: '_'.join(x), axis=1)
val_results['threshold'] = val_results[['cutoff', 'trim']].apply(lambda x: x.dropna().values[0], 1)
val_results['threshold'] = val_results['threshold'].apply(lambda x: np.nan if not x else 0.1 if type(x) == bool else x)
val_results['population'] = val_results['pop']
val_results['ngenes'] = val_results['nogenes']

test = r.drop(['method', 'ngenes'], 1)
val = val_results[test.columns]
bench = pd.concat([test, val], ignore_index=True)

groups = ['name', 'population','test', 'threshold']
diffs = bench.groupby(by=groups, as_index=True).agg(lambda x: x.iloc[0]-x.iloc[1])
eps=0.00001
diffs[diffs<=eps] = 0
diffs[diffs.isna()] = 0
diffs.describe()
#################################################################################
#################################################################################
#################################################################################




#################################################################################
########### PERMUTATOR DEBUGGER ###########
#################################################################################

# Test parameters
pops=['EUR']
tests=['eMKT']
thresholds=[[0.15]]

# Reduce data
# genes = genes.sample(frac=0.2, axis=0).sample(frac=0.2, axis=1)

# Groups and mix
groups = pd.DataFrame([x.split('_') for x in genes.columns],
                      index=genes.columns,
                      columns=['Temporal', 'Anatomical'])

mix = [['Temporal', 'Anatomical']]

# Parameters
vars_alone = True
vars_and_constant = True
reps = 10


t0=time.time()
permutator_mkt(genes, ph, groups=groups, mix=mix, reps=reps)
t=time.time()-t0
t

R = permutator_mkt(genes, ph, pops, tests, thresholds, reps, groups, mix, v=False)

#################################################################################




# r = pd.DataFrame()

# a = MKT.MKT()

# for lim, n in zip([10, 50, 100],[35, 175, 350]):
#     for pop in [['AFR'], ['EUR', 'AFR']]:
#         for t in[['eMKT']]:
#             for th in [[[0]], [[0, 0.1]]]:

                # MKT.MKT.bootstrap_lim_dft = lim
                # MKT.MKT.populations_dft = pop
                # MKT.MKT.tests_dft = t
                # MKT.MKT.thresholds_dft = th
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