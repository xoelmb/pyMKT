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
import time

#################################################################################
genes = pd.read_csv(lists_dir+'exp_aa.csv', header=[0,1], index_col=0)
genes.columns = list(map(lambda x: '_'.join(x), genes.columns.values))
ph = pd.read_csv(data_dir+'metaPops.tsv', sep='\t')
#################################################################################

genes = genes.sample(frac=0.2)


#################################################################################
############################### GENERAL DEBUGGING ###############################
#################################################################################
pops=['EUR']
tests=['eMKT']
thresholds=[[0.15]]

MKT.MKT.populations_dft = pops
MKT.MKT.tests_dft = tests
MKT.MKT.thresholds_dft = thresholds
#################################################################################
# a = MKT.MKT(genes, ph)
# t0=time.time()
# r = a.test()
# time.sleep(1)
# print(time.time()-t0)
#################################################################################

# create groups
groups = pd.DataFrame([x.split('_') for x in genes.columns],
                      index=genes.columns,
                      columns=['Temporal', 'Anatomical'])

vars_alone = True
vars_and_constant = True

reps = 2
####### INTIALIZATION #######

# Create results
RESULTS = {}
# Buffer for storing constant genes, to concat later
BUFFER_CONSTANTS = []


####### GENERAL PERMUTATION #######
print('General permutation...')
RESULTS['GRAL'] = {}

# Store contants
BUFFER_CONSTANTS.append(genes[(genes == 1).all(1)].iloc[:,[0]].rename(columns=lambda x: 'GRAL'))

# Compute VARS+CONSTANTS
if vars_and_constant:
    print('Computing GRAL ALL')
    RESULTS['GRAL']['ALL_GENES'] = MKT.MKT(genes, ph).test()

# Compute VARS_ALONE
if vars_alone:
    print('Computing GRAL VARS')
    BUFFER_VAR = [genes[(~(genes.index.isin(BUFFER_CONSTANTS[-1].index)))
                       & (genes.sum(axis=1)!=0)]]
    RESULTS['GRAL']['VAR_GENES'] = MKT.MKT(BUFFER_VAR[-1], ph).test()

# Compute permutations
print('Computing GRAL PERM')
RESULTS['GRAL']['PERM'] = MKT.MKT(genes, ph).test(permute=True, reps=reps, variable_genes=BUFFER_VAR[-1].index.to_numpy(), permute_vars_alone=vars_alone, permute_vars_and_constant=vars_and_constant)



####### PER-GROUP PERMUTATIONS #######

def perm_results_to_dict(df):
    return {val[4:]: df[df['repeat']==val] for val in df['repeat'].unique()}

if groups is not None:

    for VAR in groups.columns:
        
        print(f'\n{VAR} permutations...')

        RESULTS[VAR] = {}
        BUFFER_VAR = []
        BUFFER_PERM = []

        for VALUE in groups[VAR].unique():

            # Select data
            MATCH = genes.loc[:, genes.columns.isin(groups[groups[VAR]==VALUE].index)]
            
            # Store constants
            BUFFER_CONSTANTS.append(MATCH[(MATCH == 1).all(1)].iloc[:,[0]].rename(columns=lambda x: VALUE))
            
            # Store VARS in BUFFER
            BUFFER_VAR.append(MATCH[(~(MATCH.index.isin(BUFFER_CONSTANTS[-1].index))) & 
                                             (MATCH.sum(axis=1)!=0)])
            
            # Compute permutations of VARS and/or VARS+CONSTANT
            print(f'Computing {VAR}-{VALUE} PERM...')
            BUFFER_PERM.append(MKT.MKT(MATCH, ph).test(permute=True,
                                                       reps=reps,
                                                       variable_genes=BUFFER_VAR[-1].index.to_numpy(),
                                                       permute_vars_alone=vars_alone,
                                                       permute_vars_and_constant=vars_and_constant))

        RESULTS[VAR]['PERM'] = perm_results_to_dict(pd.concat(BUFFER_PERM, axis=0, ignore_index=True))

        # Compute results of variable genes for that VAR
        if vars_alone:
            print(f'Computing {VAR} VARS ALONE...')
            RESULTS[VAR]['VAR_GENES'] = MKT.MKT(pd.concat(BUFFER_VAR, axis=1), ph).test()

# Compute results of constant genes of all VARS and GRAL
print('Computing CONSTANTS...')
RESULTS['CONSTANTS'] = MKT.MKT(pd.concat(BUFFER_CONSTANTS, axis=1), ph).test()
