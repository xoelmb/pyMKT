import pandas as pd
import numpy as np
import MKT
from itertools import product


def permutator_mkt(genes, ph, pops=None, tests=None, thresholds=None,
                   reps=100, groups=None, mix=None, vars_alone=True, vars_and_constant=True, v=True):

    ################################
    ####### HELPER FUNCTIONS #######
    ################################

    def muter(*a):
        return

    def perm_results_to_dict(df):
        return {val[5:]: df[df['repeat']==val] for val in df['repeat'].unique()}


    def condense_mask(ls):
        if isinstance(ls, list):
            if len(ls) > 1:
                return ls[0] * condense_mask(ls[1:])
            else: return ls[0]
        else:
            return ls

    #############################
    ####### INTIALIZATION #######
    #############################

    if not v: print = muter

    # Create tuned MKT
    myMKT = MKT.MKT
    if pops is not None: myMKT.populations_dft = pops
    if tests is not None: myMKT.tests_dft = tests
    if thresholds is not None: myMKT.thresholds_dft = thresholds
    if v is not None: myMKT.verbose = v

    # Create results
    RESULTS = {}

    # Create DATABASE for GENES EXPRESSED per GROUP
    GENES_DB = {}

    # Buffer for storing constant/var genes
    BUFFER_CONSTANTS = []
    BUFFER_VAR_DB = []


    ###################################
    ####### GENERAL PERMUTATION #######
    ###################################

    print('General permutation...')
    RESULTS['GRAL'] = {}

    # Store constants
    BUFFER_CONSTANTS.append(genes[(genes == 1).all(1)].iloc[:,[0]].rename(columns=lambda x: 'GRAL'))

    # Compute VARS+CONSTANTS
    if vars_and_constant:
        print('Computing GRAL ALL')
        RESULTS['GRAL']['ALL_GENES'] = myMKT(genes, ph).test()

    # Compute VARS_ALONE
    if vars_alone:
        print('Computing GRAL VARS')
        BUFFER_VAR = [genes[(~(genes.index.isin(BUFFER_CONSTANTS[-1].index)))
                        & (genes.sum(axis=1)!=0)]]
        BUFFER_VAR_DB.append(BUFFER_VAR[-1].sum(axis=1).clip(upper=1))
        BUFFER_VAR_DB[-1].name = 'GRAL'
        RESULTS['GRAL']['VAR_GENES'] = myMKT(BUFFER_VAR[-1], ph).test()

    # Compute permutations
    print('Computing GRAL PERM')
    RESULTS['GRAL']['PERM'] = perm_results_to_dict(myMKT(genes, ph).test(permute=True, reps=reps, variable_genes=BUFFER_VAR[-1].index.to_numpy(), permute_vars_alone=vars_alone, permute_vars_and_constant=vars_and_constant))


    ######################################
    ####### PER-GROUP PERMUTATIONS #######
    ######################################

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
                BUFFER_PERM.append(myMKT(MATCH, ph).test(permute=True,
                                                         reps=reps,
                                                         variable_genes=BUFFER_VAR_DB[-1].index.to_numpy(),
                                                         permute_vars_alone=vars_alone,
                                                         permute_vars_and_constant=vars_and_constant))

            RESULTS[VAR]['PERM'] = perm_results_to_dict(pd.concat(BUFFER_PERM, axis=0, ignore_index=True))

            # Compute results of variable genes for that VAR
            if vars_alone:
                print(f'Computing {VAR} VARS ALONE...')
                RESULTS[VAR]['VAR_GENES'] = myMKT(pd.concat(BUFFER_VAR, axis=1), ph).test()


    GENES_DB['CONSTANTS'] = pd.concat(BUFFER_CONSTANTS, axis=1).fillna(0).astype(int)
    GENES_DB['VARS'] = pd.concat(BUFFER_VAR_DB, axis=1).fillna(0).astype(int)


    ################################
    ####### MIX PERMUTATIONS #######
    ################################

    if mix is not None:

        for combine in mix:

            VAR = '+'.join(combine)

            TO_COMBINE = groups[combine]
            UNIQUES = [np.unique(x) for x in TO_COMBINE.to_numpy().astype(str).T]
            
            RESULTS[VAR] = {}
            BUFFER_VAR = []
            BUFFER_PERM = []
            
            for VAL_TUPLE in product(*UNIQUES):

                VALUE = '+'.join(VAL_TUPLE)

                # Select data
                MASK = condense_mask([groups[combine[i]]==VAL_TUPLE[i] for i in range(len(combine))])
                MATCH = genes.loc[:, genes.columns.isin(groups[MASK].index)]

                if len(MATCH.columns) == 0:
                    continue
            
                # Store CONSTANTS
                BUFFER_CONSTANTS.append(GENES_DB['CONSTANTS'][[f'{var}_{val}' for var, val in zip(combine, VAL_TUPLE)]].sum(axis=1).clip(upper=1))
                BUFFER_CONSTANTS[-1].name = f'{VAR}_{VALUE}'

                # Store VARS_DB in BUFFER
                BUFFER_VAR_DB.append(GENES_DB['VARS'][[f'{var}_{val}' for var, val in zip(combine, VAL_TUPLE)]].sum(axis=1).clip(upper=1)) 
                BUFFER_VAR_DB[-1][~(BUFFER_VAR_DB[-1].index.isin(BUFFER_CONSTANTS[-1].index)) & 
                                BUFFER_VAR_DB[-1]!=0]
                BUFFER_VAR_DB[-1].name = f'{VAR}_{VALUE}'

                # Store VARS ALONE in BUFFER
                if vars_alone:
                    BUFFER_VAR.append(MATCH[MATCH.index.isin(BUFFER_VAR_DB[-1].index)])

                # Compute permutations of VARS and/or VARS+CONSTANT
                print(f'Computing {VAR}_{VALUE} PERM...')
                BUFFER_PERM.append(myMKT(MATCH, ph).test(permute=True,
                                                        reps=reps,
                                                        variable_genes=BUFFER_VAR_DB[-1].index.to_numpy(),
                                                        permute_vars_alone=vars_alone,
                                                        permute_vars_and_constant=vars_and_constant))

            RESULTS[VAR]['PERM'] = perm_results_to_dict(pd.concat(BUFFER_PERM, axis=0, ignore_index=True))

            # Compute results of variable genes for that VAR
            if vars_alone:
                print(f'Computing {VAR} VARS ALONE...')
                RESULTS[VAR]['VAR_GENES'] = myMKT(pd.concat(BUFFER_VAR, axis=1), ph).test()


    ###########################################
    ####### CONSTANT & VARS ESTIMATIONS #######
    ###########################################

    print('\nComputing CONSTANTS PER GROUP...')
    GENES_DB['CONSTANTS'] = pd.concat(BUFFER_CONSTANTS, axis=1).fillna(0).astype(int)
    RESULTS['CONSTANTS'] = myMKT(GENES_DB['CONSTANTS'], ph).test()
    print('\nComputing VARS PER GROUP...')
    GENES_DB['VARS'] = pd.concat(BUFFER_VAR_DB, axis=1).fillna(0).astype(int)
    RESULTS['VARS'] = myMKT(GENES_DB['VARS'], ph).test()


    return RESULTS



#################################################################################
############################### GENERAL DEBUGGING ###############################
#################################################################################

# root_dir = '/home/xoel/Escritorio/mastersthesis/'
# data_dir = root_dir+'data/'
# lists_dir = data_dir+'lists/'
# scripts_dir = root_dir+'scripts/'
# results_dir = root_dir+'results/'
# plots_dir = root_dir+'plots/'

# genes = pd.read_csv(lists_dir+'exp_aa.csv', header=[0,1], index_col=0)
# genes.columns = list(map(lambda x: '_'.join(x), genes.columns.values))
# ph = pd.read_csv(data_dir+'metaPops.tsv', sep='\t')

# # Reduce data
# genes = genes.sample(frac=0.2, axis=0).sample(frac=0.2, axis=1)

# # Test parameters
# pops=['EUR']
# tests=['eMKT']
# thresholds=[[0.15]]

# # Groups and mix
# groups = pd.DataFrame([x.split('_') for x in genes.columns],
#                       index=genes.columns,
#                       columns=['Temporal', 'Anatomical'])

# mix = [['Temporal', 'Anatomical']]
# # Parameters
# vars_alone = True
# vars_and_constant = True
# reps = 1


# R = permutator_mkt(genes, ph, pops, tests, thresholds, reps, mix)