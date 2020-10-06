root_dir = '/home/xoel/Escritorio/mastersthesis/'
data_dir = root_dir+'data/'
lists_dir = data_dir+'lists/'
scripts_dir = root_dir+'scripts/'
results_dir = root_dir+'results/'
plots_dir = root_dir+'plots/'

import pandas as pd
import numpy as np

genes = pd.read_csv(lists_dir+'exp_aa.csv', header=[0,1], index_col=0)
ph = pd.read_csv(data_dir+'metaPops.tsv', sep='\t')


