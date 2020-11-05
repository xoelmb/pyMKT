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
from itertools import product

#################################################################################
genes = pd.read_csv(lists_dir+'exp_aa.csv', header=[0,1], index_col=0)
genes.columns = list(map(lambda x: '_'.join(x), genes.columns.values))
ph = pd.read_csv(data_dir+'metaPops.tsv', sep='\t')
#################################################################################


#################################################################################
############################### GENERAL DEBUGGING ###############################
#################################################################################
genes = genes.sample(frac=0.2, axis=1)
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


#################################################################################
####### HELPER FUNCTIONS #######
#################################################################################

def perm_results_to_dict(df):
    return {val[4:]: df[df['repeat']==val] for val in df['repeat'].unique()}


def condense_mask(ls):
    if isinstance(ls, list):
        if len(ls) > 1:
            return ls[0] * condense_mask(ls[1:])
        else: return ls[0]
    else:
        return ls

#################################################################################
#################################################################################



#################################################################################
###### THESE ARE INPUT PARAMETERS ######
#################################################################################

# create groups
groups = pd.DataFrame([x.split('_') for x in genes.columns],
                      index=genes.columns,
                      columns=['Temporal', 'Anatomical'])

mix = [['Temporal', 'Anatomical']]

vars_alone = True
vars_and_constant = True
reps = 1

#################################################################################
#################################################################################



#################################################################################
####### INTIALIZATION #######
#################################################################################

# Create results
RESULTS = {}

# Create DATABASE for GENES
GENES_DB = {}

# Buffer for storing constant genes, to concat later
BUFFER_CONSTANTS = []
BUFFER_VAR_DB = []

#################################################################################
#################################################################################



#################################################################################
####### GENERAL PERMUTATION #######
#################################################################################

print('General permutation...')
RESULTS['GRAL'] = {}

# Store constants
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
    BUFFER_VAR_DB.append(BUFFER_VAR[-1].sum(axis=1).clip(upper=1))
    BUFFER_VAR_DB[-1].name = 'GRAL'
    RESULTS['GRAL']['VAR_GENES'] = MKT.MKT(BUFFER_VAR[-1], ph).test()

# Compute permutations
print('Computing GRAL PERM')
RESULTS['GRAL']['PERM'] = perm_results_to_dict(MKT.MKT(genes, ph).test(permute=True, reps=reps, variable_genes=BUFFER_VAR[-1].index.to_numpy(), permute_vars_alone=vars_alone, permute_vars_and_constant=vars_and_constant))

#################################################################################
#################################################################################



#################################################################################
####### PER-GROUP PERMUTATIONS #######
#################################################################################

if groups is not None:

    for VAR in groups.columns:
        
        print(f'\n{VAR} permutations...')

        RESULTS[VAR] = {}
        BUFFER_VAR = []
        BUFFER_PERM = []

        for VALUE in groups[VAR].unique():

            # Select data
            MATCH = genes.loc[:, genes.columns.isin(groups[groups[VAR]==VALUE].index)]
            
            if len(MATCH.columns) == 0:
                continue

            # Store constants
            BUFFER_CONSTANTS.append(MATCH[(MATCH == 1).all(1)].iloc[:,[0]].rename(columns=lambda x: f'{VAR}_{VALUE}'))
            
            # Store VARS and VAR_PERM in BUFFER
            BUFFER_VAR.append(MATCH[(~(MATCH.index.isin(BUFFER_CONSTANTS[-1].index))) & 
                                             (MATCH.sum(axis=1)!=0)])
            BUFFER_VAR_DB.append(BUFFER_VAR[-1].sum(axis=1).clip(upper=1))
            BUFFER_VAR_DB[-1].name = f'{VAR}_{VALUE}'
            
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

GENES_DB['CONSTANTS'] = pd.concat(BUFFER_CONSTANTS, axis=1).fillna(0).astype(int)
GENES_DB['VARS'] = pd.concat(BUFFER_VAR_DB, axis=1).fillna(0).astype(int)

#################################################################################
#################################################################################




#################################################################################
####### MIX PERMUTATIONS #######
#################################################################################
    # MIX_BUFFER = []

    # for combine in mix:

    #     MIX_BUFFER.append(groups[combine].apply(lambda x: '_'.join(x.to_list()), axis=1))
    #     MIX_BUFFER[-1].name = '_'.join(combine)
    
    # MIX_GROUPS = pd.concat(MIX_BUFFER, axis=1)

APP=[]

if mix is not None:

    for combine in mix:

        TO_COMBINE = groups[combine]
        UNIQUES = [np.unique(x) for x in TO_COMBINE.to_numpy().astype(str).T]
        
        RESULTS[VAR] = {}
        BUFFER_VAR = []
        BUFFER_PERM = []
        
        for VAL_TUPLE in product(*UNIQUES):

            # Select data
            MASK = condense_mask([groups[combine[i]]==VAL_TUPLE[i] for i in range(len(combine))])
            MATCH = genes.loc[:, genes.columns.isin(groups[MASK].index)]

            if len(MATCH.columns) == 0:
                continue
            else:
                C = combine
                V = VAL_TUPLE
                APP.append(MATCH)
            
            # Store constants
            BUFFER_CONSTANTS.append(GENES_DB['CONSTANTS'][[f'{VAR}_{VAL}'for VAR, VAL in zip(combine, VAL_TUPLE)]].sum(axis=1).clip(upper=1))
            BUFFER_CONSTANTS[-1].name = "+".join(combine)

            # Store VARS in BUFFER
            BUFFER_VAR.append(GENES_DB['VARS'][[f'{VAR}_{VAL}'for VAR, VAL in zip(combine, VAL_TUPLE)]].sum(axis=1).clip(upper=1)) 
            BUFFER_VAR[-1][~(BUFFER_VAR[-1].index.isin(BUFFER_CONSTANTS[-1].index))]
            BUFFER_VAR[-1].name = "+".join(combine)

            # Compute permutations of VARS and/or VARS+CONSTANT
            print(f'Computing {combine}-{VAL_TUPLE} PERM...')
            BUFFER_PERM.append(MKT.MKT(MATCH, ph).test(permute=True,
                                                       reps=reps,
                                                       variable_genes=BUFFER_VAR[-1].index.to_numpy(),
                                                       permute_vars_alone=vars_alone,
                                                       permute_vars_and_constant=vars_and_constant))

        RESULTS[VAR]['PERM'] = perm_results_to_dict(pd.concat(BUFFER_PERM, axis=0, ignore_index=True).fillna(0).astype(int))

        # Compute results of variable genes for that VAR
        if vars_alone:
            print(f'Computing {VAR} VARS ALONE...')
            RESULTS[VAR]['VAR_GENES'] = MKT.MKT(pd.concat(BUFFER_VAR, axis=1), ph).test()

#################################################################################
#################################################################################



#################################################################################
####### CONSTANT & VARS ESTIMATIONS #######
#################################################################################

print('\nComputing CONSTANTS...')
GENES_DB['CONSTANTS'] = pd.concat(BUFFER_CONSTANTS, axis=1).fillna(0).astype(int)
RESULTS['CONSTANTS'] = MKT.MKT(GENES_DB['CONSTANTS'], ph).test()
GENES_DB['VARS'] = pd.concat(BUFFER_VAR_DB, axis=1).fillna(0).astype(int)
RESULTS['VARS'] = MKT.MKT(GENES_DB['VARS'], ph).test()

#################################################################################
#################################################################################



#################################################################################
#################################################################################
