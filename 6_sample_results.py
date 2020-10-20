#!/usr/bin/python

# Esse arquivo é para comparar, no sample que escolhi, os modelos e métricas.
# Em especial, checar se a solução correta tem menor distância.
# Caso contrário, teremos problemas em melhorar os métodos.

import xml.etree.ElementTree as ET
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from nltk.corpus import stopwords
brown_ic = wordnet_ic.ic('ic-brown.dat')
from nltk import FreqDist
import numpy as np
import networkx as nx
import pickle
import re
import pandas as pd

# para salvar os gtsp preparados
import os
import gzip
from collections import defaultdict
from time import time

# Taking from the step2 models
from step2_heuristic import degree, mfs, gold_standard
from step2_tspaco import single_aco, stochastic_aco, aco
from step2_glns import glns
from step2_dijkstra import dijkstra_frasal, pop2010

gpickle_folder = './data/sample/'

all_gold_loc = './data/WSD_Unified_Evaluation_Datasets/ALL/ALL.gold.key.txt'
gold = {}
with open(all_gold_loc, 'r') as f:
    for line in f.readlines():
        gold[line.split()[0]] = line.split()[1:]
gold

# G = nx.read_gpickle('data/sample/semeval2013.d012.s010.gpickle')
# G.edges()[('semeval2013.d012.s010.t004.c007', 'semeval2013.d012.s010.t000.c001')]

# Functions
def path_length(G, node_ids, measure='sim_jcn_log'):
    # node_ids = pred
    l = 0
    for i in range(len(node_ids)-1):
        # i=0
        u, v = node_ids[i], node_ids[i+1]
        try:
            # w = G.edges(data=measure)[(u, v)]
            w = G.edges[u, v][measure]
        except:
            print('caminho desconectado')
            w = np.inf
        l += w
    return l

# Analisando as respostas do gold_standard
sent_list = os.listdir(gpickle_folder)
sent_list = [s for s in sent_list if s[-8:] == '.gpickle']


def measure_sentence(sent_id, model, gpickle_folder, params={}):
    if 'measure' not in params:
        params['measure'] = 'dist'
    # sent_id = 'semeval2013.d012.s010'
    # model = dijkstra_frasal
    gpickle_file = gpickle_folder + sent_id + '.gpickle'
    G = nx.read_gpickle(gpickle_file)
    # list(G)
    # print(sent_id)
    # params={'measure': 'dist_als'}
    # 'semeval2013.d012.s010.t002.c000.copy'
    pred = model(G, params)
    l = path_length(G, pred, measure=params['measure'])
    solution = {}
    for p in pred:
        # p = pred[0]
        inst_id = p[:-5]
        keys = []
        for lemma in wn.synset(G.nodes()[p]['synset']).lemmas():
            keys.append(lemma.key())
        solution[inst_id] = keys
    del(G)
    return l, solution
# measure_sentence(sent_id, model)

def measure_solution(model, gpickle_folder, n=-1, params={}):
    # model = mfs
    # params = {'measure': 'dist_als'}
    sent_list = os.listdir(gpickle_folder)
    sent_list = [s for s in sent_list if s[-8:] == '.gpickle']
    sent_list = sent_list[:n]
    sent_result = {}
    solutions = {}
    for file in sent_list:
        # file = sent_list[0]
        sent_id = file[:-8]
        l, solution = measure_sentence(sent_id, model, gpickle_folder, params)
        sent_result[file] = l
        solutions[sent_id] = solution
    avg_l = np.mean([sent for sent in sent_result.values()])
    # sum_acc = sum_tp / (sum_tp + sum_fp)
    print('average length: ' + str(avg_l))
    return avg_l, solutions


# Comparing models
def run_models(gpickle_folder, n=-1):
    models = ['mfs',
              'gold_standard',
              'degree',
              'pop2010',
              'dijkstra_frasal',
              'aco']
    results = {}
    solutions_df = pd.DataFrame()
    for model_name in models:
        start = time()
        print('model: {}'.format(model_name))
        # model_name = 'gold'
        # n = -1
        if model_name == 'aco':
            measures = ['dist_sim_jcn_ratio', 'dist_sim_jcn_log', 'dist_sim_als']
            model = single_aco
            params={'iter':10}
        else:
            measures = ['sim_jcn_ratio', 'sim_jcn_log', 'sim_als']
            model = eval(model_name)
            params={}
        for m in measures:
            print(m)
            params['measure'] = m
            l, solutions = measure_solution(model, gpickle_folder, n, params)
            results[model_name + '_' + m] = l
        end = time()
        temp = pd.Series()
        for s in solutions:
            temp = temp.append(pd.Series(solutions[s]))
        solutions_df[model_name] = temp
        print('demorou {} segundos total'.format(int(end-start)))
    return results, solutions_df


def save_results(sol_df, gpickle_folder):
    sol_df.columns
    sol_df['gold'] = pd.Series(gold)

    def compare_columns(d, c1, c2):
        try:
            return len([v for v in d[c1] if v in d[c2]]) >= 1
        except:
            return False
    sol_df['mfs_gold'] = sol_df.apply(compare_columns, axis=1, args=('mfs', 'gold'))
    sol_df['aco_gold'] = sol_df.apply(compare_columns, axis=1, args=('aco', 'gold'))
    sol_df['deg_gold'] = sol_df.apply(compare_columns, axis=1, args=('degree', 'gold'))
    sol_df['glns_gold'] = sol_df.apply(compare_columns, axis=1, args=('glns', 'gold'))
    sol_df['aco_mfs'] = sol_df.apply(compare_columns, axis=1, args=('aco', 'mfs'))
    sol_df['deg_mfs'] = sol_df.apply(compare_columns, axis=1, args=('deg', 'mfs'))
    sol_df['glns_mfs'] = sol_df.apply(compare_columns, axis=1, args=('glns', 'mfs'))
    sol_df['aco_glns'] = sol_df.apply(compare_columns, axis=1, args=('aco', 'glns'))

    with open(gpickle_folder+'/results.pickle', 'wb') as f:
        pickle.dump(sol_df, f)
    print('concluído!')
    return None


def all_models(n=50):
    # results, sol_df = run_models(models=[degree, mfs, aco, glns])
    for special_folder in ['jcn+lesk_ratio',
                           'jcn+lesk_log', 'al_saiagh']:
        print('###' + special_folder)
        gpickle_folder = './data/' + special_folder + '/'
        results, sol_df = run_models(gpickle_folder, n=n)
        # save_results(sol_df, gpickle_folder)
    return None
all_models(n=-1)
#
#
# ### Trabalhando os melhores resultados
# with open('./data/resultados.pickle', 'rb') as f:
#     sol_df = pickle.load(f)
# print('concluído!')
# sol_df
# sol_df.columns
# pd.crosstab(sol_df['mfs_gold'], sol_df['aco_gold'])
# pd.crosstab(sol_df['mfs_gold'], sol_df['deg_gold'])

### Comparando stochastic_aco com single_aco
# Para ser justo, compararei 10x10, 100x1 e 20x5
