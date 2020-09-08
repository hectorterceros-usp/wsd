#!/usr/bin/python

# Esse arquivo é para transformar um problema WSD em GTSP.
# Essa funções servem para qualquer das soluções disponíveis.
# Em seguida, de acordo com a solução escolhida, seguir para seu script.

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
from step2_heuristic import degree, mfs
from step2_tspaco import single_aco, stochastic_aco, aco
from step2_glns import glns

gpickle_folder = './data/jcn+lesk_ratio/'

all_gold_loc = './data/WSD_Unified_Evaluation_Datasets/ALL/ALL.gold.key.txt'
gold = {}
with open(all_gold_loc, 'r') as f:
    for line in f.readlines():
        gold[line.split()[0]] = line.split()[1:]
gold


# Functions
def score_sentence(sent_id, model, gpickle_folder, params={}):
    # sent_id = 'senseval3.d000.s092'
    # model = stochastic_aco
    gpickle_file = gpickle_folder + sent_id + '.gpickle'
    G = nx.read_gpickle(gpickle_file)
    # list(G)
    # print(sent_id)
    pred = model(G, params)
    tp, fp = 0, 0
    solution = {}
    for p in pred:
        # p = pred[0]
        inst_id = p[:-5]
        keys = []
        for l in wn.synset(G.nodes()[p]['synset']).lemmas():
            keys.append(l.key())
        if len(set(gold[inst_id]) & set(keys)) > 0:
            tp += 1
        else:
            fp += 1
        solution[inst_id] = keys
    del(G)
    return tp, fp, solution
# score_sentence(sent_id, model)

def score_solution(model, gpickle_folder, n=-1, params={}):
    sent_list = os.listdir(gpickle_folder)
    sent_list = [s for s in sent_list if s[-8:] == '.gpickle']
    sent_result = {}
    sent_list = sent_list[:n]
    solutions = {}
    for file in sent_list:
        # file = sent_list[0]
        sent_id = file[:-8]
        tp, fp, solution = score_sentence(sent_id, model, gpickle_folder, params)
        if tp + fp == 0:
            continue
        acc = tp / (tp + fp)
        sent_result[sent_id] = {'tp': tp, 'fp': fp, 'acc': acc}
        solutions[sent_id] = solution
    sum_tp = sum([sent['tp'] for sent in sent_result.values()])
    sum_fp = sum([sent['fp'] for sent in sent_result.values()])
    avg_acc = np.mean([sent['acc'] for sent in sent_result.values()])
    # sum_acc = sum_tp / (sum_tp + sum_fp)
    print(sum_tp, sum_fp, avg_acc)
    return (sum_tp, sum_fp, avg_acc), solutions


# Comparing models
def run_models(gpickle_folder, n=-1, models=['mfs', 'degree', 'aco5', 'aco25', 'aco100']):
    results = {}
    solutions_df = pd.DataFrame()
    for model in models:
        start = time()
        print('model: {}'.format(model))
        # model = 'stochastic_aco100'
        # n = -1
        if model == 'aco5':
            r, solutions = score_solution(single_aco, gpickle_folder, n, params={'iter':5})
        elif model == 'aco25':
            r, solutions = score_solution(single_aco, gpickle_folder, n, params={'iter':25})
        elif model == 'aco100':
            r, solutions = score_solution(single_aco, gpickle_folder, n, params={'iter':100})
        else:
            r, solutions = score_solution(eval(model), gpickle_folder, n)
        results[model] = r
        end = time()
        temp = pd.Series()
        for s in solutions:
            temp = temp.append(pd.Series(solutions[s]))
        solutions_df[model] = temp
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
    # sol_df['glns_gold'] = sol_df.apply(compare_columns, axis=1, args=('glns', 'gold'))
    sol_df['aco_mfs'] = sol_df.apply(compare_columns, axis=1, args=('aco', 'mfs'))
    sol_df['deg_mfs'] = sol_df.apply(compare_columns, axis=1, args=('deg', 'mfs'))
    # sol_df['glns_mfs'] = sol_df.apply(compare_columns, axis=1, args=('glns', 'mfs'))
    # sol_df['aco_glns'] = sol_df.apply(compare_columns, axis=1, args=('aco', 'glns'))

    with open(gpickle_folder+'/results.pickle', 'wb') as f:
        pickle.dump(sol_df, f)
    print('concluído!')
    return None


def all_models():
    # results, sol_df = run_models(models=[degree, mfs, aco, glns])
    for special_folder in ['jcn+lesk_ratio', 'jcn+lesk_ratio_small',
                           'jcn+lesk_log', 'lesk', 'al_saiagh']:
        print('###' + special_folder)
        gpickle_folder = './data/' + special_folder + '/'
        results, sol_df = run_models(gpickle_folder, n=50)
        save_results(sol_df, gpickle_folder)
    return None

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
