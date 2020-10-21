import tracemalloc

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
from math import log10
from memorizer import Memorizer


genes = pd.read_csv(lists_dir+'exp_aa.csv', header=[0,1], index_col=0)
genes.columns = list(map(lambda x: '_'.join(x), genes.columns.values))
ph = pd.read_csv(data_dir+'metaPops.tsv', sep='\t')


r = pd.DataFrame()

MKT.MKT.populations_dft = ['AFR']
MKT.MKT.tests_dft = ['eMKT']
MKT.MKT.thresholds_dft = [[0]]

def get_input_data(df, n=50, cols_range=(1,200), rows_range=(1000, 6000)):

    dfs = []

    i_df = df.copy()
    for _ in range(n):
        n_rows = np.random.randint(*rows_range)
        n_cols = np.random.randint(*cols_range)
        i_df = i_df.sample(n_rows, axis=0, replace=True).sample(n_cols, axis=1, replace=True)
        dfs.append(i_df)
    
    return dfs

n=10

dfs = get_input_data(genes, n=n)

res_dicts = []

for i, df in zip(list(range(len(dfs))), dfs):

    print(f'{"#"*i+" "}{i}/{len(dfs)}{" "+"#"*(len(dfs)-i)}')

    res = {}
    res['reps'] = np.random.randint(200, 1000)
    # res['reps'] = 599
    res['genes'] = df.sum(axis=0).mean()
    res['samples'] = len(df.columns.values)
    res['factor'] = res['reps'] * res['genes'] * res['samples']
    while log10(res['factor']) > 10.2:
        res['reps'] = np.random.randint(10, 1000)
        res['factor'] = res['reps'] * res['genes'] * res['samples']

 
    # tracemalloc.start()
    memory = Memorizer()

    a = MKT.MKT(df, ph)
    a.bootstrap(res['reps'], max_ram=13e9)
    # r = pd.concat([r, a.bootstrap(res['reps'], max_ram=5*10**9)], axis=0)

    # tracemalloc.stop()
    mem = memory.stop()
    
    res['lowest'], res['peak'], res['mean'] = min(mem), max(mem), mem.mean()

    print(f"# REPS: {res['reps']}  GENES: {int(res['genes'])}  SAMPLES: {res['samples']}  FACTOR: {int(log10(res['factor']))}  MEM: {round(res['peak'] / 10**9, 2)} GB", end='\n\n')
    res_dicts.append(res)


results = pd.DataFrame(res_dicts)
tresults = results[results['peak'] > 10**9]


from sklearn.linear_model import LinearRegression
x=np.log10(tresults['factor'].to_numpy()).reshape((len(tresults['factor']), 1))
y=np.log10(tresults['peak'].to_numpy())
modelf = LinearRegression().fit(x, y)

print(modelf.score(x, y))

print(modelf.coef_[0], modelf.intercept_)

coef, inte = modelf.coef_[0], modelf.intercept_


import matplotlib.pyplot as plt
import seaborn as sns
sns.lineplot(tresults['factor'], tresults['peak'])
sns.lineplot(tresults['factor'], [10**(coef*f+inte) for f in np.log10(tresults['factor'])], c='red')
plt.show()
plt.close()

sns.lineplot(np.log10(tresults['factor']), np.log10(tresults['peak']))
sns.lineplot(np.log10(tresults['factor']), [coef*f+inte for f in np.log10(tresults['factor'])], c='red')
plt.show()
plt.close()
