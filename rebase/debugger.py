-
t0=time.time()
r = a.test()
time.sleep(1)
print(time.time()-t0)

t0=time.time()
r = a.test(bootstrap=True, reps=10)
time.sleep(1)
print(time.time()-t0)
#################################################################################



#################################################################################
#################### CHECK HOW CHUNKSIZE AFFECTS PERFORMANCE ####################
#################################################################################
import time
import matplotlib.pyplot as plt

a = MKT.MKT(genes, ph, frac=0.1)
r = []
pos = [1,15,25,50, 100, 500]
reps = [1, 50, 100, 500, 1000]
for c in pos:
    print(c)
    for rep in reps:
        print(rep)
        t0=time.time()
        a.test(bootstrap=True, reps=rep, c=c)
        r.append(dict(r=rep, c=c, t=time.time()-t0))

r = pd.DataFrame(r)
plt.plot(r['r'], r['t'])

t0=time.time()
a.test(populations=pops, tests=tests, thresholds=thresholds, c=100)
time.time()-t0
#################################################################################
#################################################################################
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