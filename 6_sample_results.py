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
import sys
import pickle

# para salvar os gtsp preparados
import os
import gzip
from collections import defaultdict
from time import time

# Taking from the step2 models
from step2_heuristic import degree, mfs, gold_standard
from step2_tspaco import single_aco, stochastic_aco, aco
from step2_glns import glns
from step2_dijkstra import dijkstra_frasal, dijkstra_pop2010

gpickle_folder = './data/sample/'

all_gold_loc = './data/WSD_Unified_Evaluation_Datasets/ALL/ALL.gold.key.txt'
gold = {}
with open(all_gold_loc, 'r') as f:
    for line in f.readlines():
        gold[line.split()[0]] = line.split()[1:]
gold

pred_dict = {}

# G = nx.read_gpickle('data/sample/semeval2013.d012.s010.gpickle')
# G.edges()[('semeval2013.d012.s010.t004.c007', 'semeval2013.d012.s010.t000.c001')]

# Para mapear as respostas de cada modelo, checando se Dijkstra está
# desviando de Degree

# Functions
def path_length(G, node_ids, measure='sim_jcn_log'):
    # node_ids = pred
    if measure[:5] != 'dist_':
        measure = 'dist_'+measure
    try:
        l = G.edges[node_ids[-1], node_ids[0]][measure]
    except:
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
    # também estou juntando o que preciso do score_sentence()
    # assim tenho as duas informações aqui
    if 'measure' not in params:
        params['measure'] = 'dist_sim_jcn_log'
    # sent_id = 'semeval2013.d012.s010'
    # model = gold_standard
    gpickle_file = gpickle_folder + sent_id + '.gpickle'
    G = nx.read_gpickle(gpickle_file)
    # len(G)
    # print(G.edges(data='sim_als'))
    # print(sent_id)
    # params={'measure': 'dist_sim_jcn_ratio'}
    # 'semeval2013.d012.s010.t002.c000.copy'
    pred = model(G, params)
    pred_name = model.__name__ + '__' + sent_id
    pred_dict[pred_name] = pred
    tp, fp = 0, 0
    l = path_length(G, pred, measure=params['measure'])
    solution = {}
    for p in pred:
        # p = pred[0]
        inst_id = p[:-5]
        keys = []
        for lemma in wn.synset(G.nodes()[p]['synset']).lemmas():
            keys.append(lemma.key())
        if len(set(gold[inst_id]) & set(keys)) > 0:
            tp += 1
        else:
            fp += 1
        solution[inst_id] = keys
    del(G)
    # acc = tp / (tp + fp)
    return l, solution, tp, fp
# measure_sentence(sent_id, model)

def measure_solution(model, gpickle_folder, n=-1, params={}):
    # model = mfs
    # params = {'measure': 'dist_als'}
    sent_list = os.listdir(gpickle_folder)
    sent_list = [s for s in sent_list if s[-8:] == '.gpickle']
    sent_list = sent_list[:n]
    sent_result = {}
    solutions = {}
    accs = {}
    sum_tp, sum_fp = 0, 0
    for file in sent_list:
        # file = sent_list[0]
        sent_id = file[:-8]
        l, solution, tp, fp = measure_sentence(sent_id, model, gpickle_folder, params)
        sent_result[file] = l
        solutions[sent_id] = solution
        sum_tp += tp
        sum_fp += fp
        accs[file] = tp / (tp + fp)
    acc = sum_tp/(sum_tp + sum_fp)
    avg_l = np.mean([sent for sent in sent_result.values()])
    # sum_acc = sum_tp / (sum_tp + sum_fp)
    print('avg len: ' + str(avg_l) + '; avg acc: ' + str(acc))
    return sent_result, solutions, accs


# Comparing models
def run_models(gpickle_folder, n=-1, complexity=100):
    models = ['gold_standard',
              'mfs',
              'degree',
              'dijkstra_frasal',
              'dijkstra_pop2010',
              'single_aco',
              # 'stochastic_aco',
              ]
    results = {}
    all_accs = {}
    solutions_df = pd.DataFrame()
    complexity = complexity
    for model_name in models:
        start = time()
        print('model: {}'.format(model_name))
        # model_name = 'gold'
        # n = -1
        # measures = ['dist_sim_jcn_ratio', 'dist_sim_jcn_log']
        measures = ['dist_sim_jcn_log']
        params={}
        model = eval(model_name)
        if model_name in ['single_aco', 'degree', 'gold_standard']:
            measures = ['sim_jcn_log']
            params['iter'] = 10 * complexity
        if model_name in ['dijkstra_pop2010']:
            params['generations'] = complexity
        if model_name in ['stochastic_aco']:
            params['iter'] = complexity
            params['runs'] = 10
        results[model_name] = {}
        all_accs[model_name] = {}
        for m in measures:
            params['measure'] = m
            print(params)
            l, solutions, accs = measure_solution(model, gpickle_folder, n, params)
            results[model_name][m] = l
            all_accs[model_name][m] = accs
        end = time()
        temp = pd.Series()
        for s in solutions:
            temp = temp.append(pd.Series(solutions[s]))
        solutions_df[model_name] = temp
        print('demorou {} segundos total'.format(int(end-start)))
    return results, solutions_df, all_accs
# inverter os dicts
def inverte(r):
    alt_r = {}
    for (k, i) in r.items():
        # print(k, i)
        for (kk, ii) in i.items():
            # print('dentro', kk, ii)
            tipo = kk.split('_')[-1]
            if tipo not in alt_r:
                alt_r[tipo] = {k: ii}
            else:
                alt_r[tipo][k] = ii
    return alt_r
# alt_r = inverte(r)

# Rodando tudo
def main():
    try:
        complexity = int(sys.argv[1])
    except:
        complexity = 100
    r, s, a = run_models(gpickle_folder, complexity=complexity)
    with open('data/results/sample_solutions.pickle', 'wb') as f:
        pickle.dump(s, f)

    log_df = pd.DataFrame(inverte(r)['log']).join(
        pd.DataFrame(inverte(a)['log']), lsuffix='_dist', rsuffix='_acc')
    # ratio_df = pd.DataFrame(inverte(r)['ratio']).join(
    #     pd.DataFrame(inverte(a)['ratio']), lsuffix='_dist', rsuffix='_acc')

    # para salvar esses resultados de forma mais organizada e inteligente
    cols = log_df.columns.tolist()
    cols.sort()

    excel = pd.ExcelWriter('data/results/comparação_modelos.xls')
    log_df[cols].to_excel(excel, sheet_name='log')
    # ratio_df[cols].to_excel(excel, sheet_name='ratio')
    excel.close()

if __name__ == "__main__":
    main()
    with open('data/results/preds.pickle', 'wb') as f:
        pickle.dump(pred_dict, f)
