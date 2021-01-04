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
import matplotlib.pyplot as plt
import sys
import pickle

# para salvar os gtsp preparados
import os
import gzip
from collections import defaultdict
from time import time

# Taking from the step2 models
from step2_heuristic import degree, degree_dist, mfs, gold_standard
from step2_tspaco import single_aco, stochastic_aco, aco
from step2_glns import glns
from step2_dijkstra import dijkstra_frasal, dijkstra_pop2010

folder = './data/sample/'

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
    if measure[:4] == 'sim_':
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
# sent_list = os.listdir(gpickle_folder)
# sent_list = [s for s in sent_list if s[-8:] == '.gpickle']


# measure_sentence(sent_id, model, gpickle_folder)
# measure_sentence('semeval2007.d000.s000', degree_dist, gpickle_folder,params={'measure': 'direct_dist_sim_lesk'})
def measure_sentence(sent_id, model, G, params={}):
    # também estou juntando o que preciso do score_sentence()
    # assim tenho as duas informações aqui
    if 'measure' not in params:
        params['measure'] = 'dist_sim_jcn_log'
    # sent_id = 'semeval2013.d006.s024'
    # model = dijkstra_frasal
    try:
        folder = params['folder']
    except:
        folder='data/sample/'
    with open(folder + 'all_graphs.pickle', 'rb') as f:
        graph_dict = pickle.load(f)
    # G = graph_dict[sent_id]
    n = len(G)
    # print(G.edges(data='direct_dist_sim_jcn_log'))
    # params['measure'] = 'direct_dist_sim_jcn_log'
    # print(sent_id)
    # params={'measure': 'direct_dist_sim_jcn_log'}
    # 'semeval2013.d012.s010.t002.c000.copy'
    pred = model(G, params)
    pred_name = model.__name__ + '__' + sent_id
    pred_dict[pred_name] = pred
    tp, fp = 0, 0
    l = path_length(G, node_ids=pred, measure=params['measure'])
    solution = {}
    prediction = {}
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
        prediction[inst_id] = p
    del(G)
    # acc = tp / (tp + fp)
    return l, prediction, tp, fp, n
# measure_sentence(sent_id, model)

def measure_solution(model, graph_dict, n=-1, params={}):
    # model = mfs
    # params = {'measure': 'dist_als'}
    sent_result = {}
    solutions = {}
    accs = {}
    sum_tp, sum_fp = 0, 0
    for sent_id in graph_dict:
        # file = sent_list[0]
        G = graph_dict[sent_id]
        l, solution, tp, fp, n = measure_sentence(sent_id, model, G, params)
        name = sent_id + '__' + str(tp+fp) + '_' + str(n)
        sent_result[name] = l
        solutions[sent_id] = solution
        sum_tp += tp
        sum_fp += fp
        accs[name] = tp / (tp + fp)
    acc = sum_tp/(sum_tp + sum_fp)
    avg_l = np.mean([sent for sent in sent_result.values()])
    # sum_acc = sum_tp / (sum_tp + sum_fp)
    print('avg len: ' + str(avg_l) + '; avg acc: ' + str(acc))
    return sent_result, solutions, accs


# Comparing models
def run_models(graph_dict, n=-1, complexity=100, folder='data/sample/'):
    models = ['gold_standard',
              'mfs',
              'degree',
              # 'degree_dist',
              'dijkstra_frasal',
              # 'dijkstra_pop2010',
              # 'single_aco',
              # 'stochastic_aco',
              ]
    results = {}
    all_accs = {}
    # solutions_df = pd.DataFrame()
    solutions_dfs = {}
    for model_name in models:
        start = time()
        print('model: {}'.format(model_name))
        # model_name = 'gold_standard'
        # n = -1
        # measures = ['dist_sim_jcn_ratio', 'dist_sim_jcn_log']
        measures = ['direct_dist_sim_jcn_log', 'direct_dist_sim_lesk',
                    'normalized_dist_sim_jcn_log', 'normalized_dist_sim_lesk','dist_sim_jcn_log', 'dist_sim_lesk',
                    ]
        params={'folder': folder}
        model = eval(model_name)
        if model_name in ['single_aco', 'degree']:
            measures = ['sim_jcn_log', 'sim_lesk']
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
            solutions_name = model_name + '__' + m
            l, solutions, accs = measure_solution(model, graph_dict, n, params)
            results[model_name][m] = l
            all_accs[model_name][m] = accs
            temp = pd.Series()
            for s in solutions:
                temp = temp.append(pd.Series(solutions[s]))
            if m not in solutions_dfs:
                solutions_dfs[m] = pd.DataFrame()
            solutions_dfs[m][model_name] = temp
        end = time()
        print('demorou {} segundos total'.format(int(end-start)))
    # solutions_df.columns[0]
    for m in solutions_dfs:
        solutions_dfs[m]['id'] = solutions_dfs[m][solutions_dfs[m].columns[0]].apply(lambda x: x[:-10])
    return results, solutions_dfs, all_accs

# inverter os dicts
def inverte(r):
    alt_r = {}
    for (k, i) in r.items():
        # print(k, i)
        for (kk, ii) in i.items():
            # print('dentro', kk, ii)
            # tipo = kk.split('_')[-1]
            tipo = kk
            if tipo not in alt_r:
                alt_r[tipo] = {k: ii}
            else:
                alt_r[tipo][k] = ii
    return alt_r
# alt_r = inverte(r)

# Rodando tudo
def rodando_tudo(folder='data/sample/'):
    with open(folder + 'all_graphs.pickle', 'rb') as f:
        graph_dict = pickle.load(f)
    try:
        complexity = int(sys.argv[1])
    except:
        complexity = 100
    r, s, a = run_models(graph_dict, complexity=complexity, folder=folder)
    with open(folder + 'all_solutions.pickle', 'wb') as f:
        pickle.dump(s, f)

    if False:
        # inverte(r)
        log_df = pd.DataFrame(inverte(r)['direct_dist_sim_jcn_log']).join(
            pd.DataFrame(inverte(a)['direct_dist_sim_jcn_log']), lsuffix='_dist', rsuffix='_acc')
        log_df['k'] = [int(v.split('_')[-2]) for v in list(log_df.index)]
        log_df = log_df.loc[log_df['k'] > 1].copy()
        gold_df = log_df[['gold_standard_dist', 'dijkstra_frasal_dist', 'dijkstra_frasal_acc']].copy()
        gold_df['ratio'] = gold_df['gold_standard_dist']/gold_df['dijkstra_frasal_dist']
        gold_df['n'] = [int(v.split('_')[-1]) for v in list(gold_df.index)]
        gold_df['k'] = [int(v.split('_')[-2]) for v in list(gold_df.index)]
        plt.figure()
        plt.scatter(gold_df['k'], gold_df['dijkstra_frasal_dist'], alpha=0.3, c='r')
        plt.scatter(gold_df['k'], gold_df['gold_standard_dist'], alpha=0.3)
        plt.title('Distância total do circuito do gabarito por número de clusters')
        plt.ylabel('Distância total (1/sim)')
        plt.xlabel('Número de clusters (k)')
        plt.legend(['dijkstra', 'gold_standard'])
        # plt.scatter(gold_df['k'], gold_df['gold_standard_dist'], color='r')
        # plt.show()
        plt.savefig('data/plots/direct_k.png')
        plt.close()

        plt.figure()
        plt.scatter(gold_df['n'], gold_df['dijkstra_frasal_dist'], alpha=0.3, c='r')
        plt.scatter(gold_df['n'], gold_df['gold_standard_dist'], alpha=0.3)
        plt.title('Distância total do circuito do gabarito por número de vértices')
        plt.ylabel('Distância total (1/sim)')
        plt.xlabel('Número de vértices (n)')
        plt.legend(['dijkstra', 'gold_standard'])
        # plt.scatter(gold_df['k'], gold_df['gold_standard_dist'], color='r')
        plt.savefig('data/plots/direct_n.png')
        plt.close()

        plt.figure()
        plt.scatter(gold_df['n'], gold_df['dijkstra_frasal_dist'], alpha=0.3, c='r')
        plt.scatter(gold_df['n'], gold_df['gold_standard_dist'], alpha=0.3)
        plt.title('Distância média do circuito do gabarito por número de vértices')
        plt.ylabel('Distância total (k/sim)')
        plt.xlabel('Número de vértices (n)')
        plt.legend(['dijkstra', 'gold_standard'])
        # plt.scatter(gold_df['k'], gold_df['gold_standard_dist'], color='r')
        plt.savefig('data/plots/normalized_n.png')
        plt.close()
        # ratio_df = pd.DataFrame(inverte(r)['ratio']).join(
        #     pd.DataFrame(inverte(a)['ratio']), lsuffix='_dist', rsuffix='_acc')

        # relação entre dist e acurácia
        gold_df = log_df[['gold_standard_dist', 'dijkstra_frasal_dist', 'dijkstra_frasal_acc']].copy()
        gold_df['ratio'] = gold_df['gold_standard_dist']/gold_df['dijkstra_frasal_dist']
        gold_df['n'] = [int(v.split('_')[-1]) for v in list(gold_df.index)]
        gold_df['k'] = [int(v.split('_')[-2]) for v in list(gold_df.index)]
        gold_df['ratio_cut'] = (gold_df['ratio']/0.5).apply(int)*0.5
        plt.figure()
        plt.scatter(gold_df['ratio'], gold_df['dijkstra_frasal_acc'], alpha=0.3, c='r')
        medias = gold_df.groupby('ratio_cut')['dijkstra_frasal_acc'].agg('mean')
        plt.plot(medias, c='b')
        # plt.scatter(gold_df['k'], gold_df['gold_standard_dist'], alpha=0.3)
        plt.title('Dispersão da acurácia frente à diferença entre o gold standard e menor caminho')
        plt.ylabel('Acurácia do menor caminho (Dijkstra)')
        plt.xlabel('Razão entre o comprimento do Gold Standard e o Dijkstra')
        plt.legend(['média', 'dados'])
        # plt.scatter(gold_df['k'], gold_df['gold_standard_dist'], color='r')
        # plt.show()
        plt.savefig('data/plots/acc_dist.png')
        plt.close()

    # para salvar esses resultados de forma mais organizada e inteligente

    log_df = pd.DataFrame(inverte(r)['direct_dist_sim_jcn_log']).join(
        pd.DataFrame(inverte(a)['direct_dist_sim_jcn_log']), lsuffix='_dist', rsuffix='_acc')
    cols = log_df.columns.tolist()
    cols.sort()

    excel = pd.ExcelWriter('data/results/comparação_modelos_all.xls')
    log_df[cols].to_excel(excel, sheet_name='log')
    # ratio_df[cols].to_excel(excel, sheet_name='ratio')
    excel.close()


def main(argv):
    try:
        folder = argv[1]
        print('usando pasta', folder)
    except:
        folder = 'data/public/'
    start = time()
    rodando_tudo(folder)
    end = time()
    print('tempo total: ', (end-start)/60, ' minutos')

if __name__ == "__main__":
    main(sys.argv)
    with open('data/results/preds.pickle', 'wb') as f:
        pickle.dump(pred_dict, f)

# print(s['degree'] == s['degree_dist'])
