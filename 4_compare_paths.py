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
    steps = []
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
        steps.append(w)
    return l, steps

# # Analisando as respostas do gold_standard
# sent_list = os.listdir(gpickle_folder)
# sent_list = [s for s in sent_list if s[-8:] == '.gpickle']

def mode_shortest(L):
    # L = shortest['p']
    resp = []
    n_words = len(L[0])
    for k in range(n_words):
        local = [l[k] for l in L]
        resp.append(FreqDist(local).max())
    return resp

# measure_sentence(sent_id, model, gpickle_folder)
# measure_sentence('semeval2007.d000.s000', degree_dist, gpickle_folder,params={'measure': 'direct_dist_sim_lesk'})
def plot_all_paths(sent_id, global_solution_df, G, params={}, plot=False):
    # sent_id = 'semeval2015.d001.s025'
    # sent_id = 'semeval2015.d001.s049'
    # G = graph_dict[sent_id]
    # global_solution_df = global_solution_df[measure]
    # params={'measure': measure}
    # len(global_solution_df)
    solution_df = global_solution_df.loc[global_solution_df ['id'] == sent_id]
    n = len(G)
    ids = {k: v for k, v in G.nodes(data='id')}
    unique_ids = list(set(ids.values()))
    unique_ids.sort()
    all_paths = [[k] for k in ids if ids[k] == unique_ids[0]]
    for i in unique_ids[1:]:
        # i = unique_ids[0]
        new_values = [k for k in ids if ids[k] == i]
        new_paths = []
        for v in new_values:
            for p in all_paths:
                new_path = p + [v]
                new_paths.append(new_path)
        all_paths = new_paths
    # params['measure'] = 'direct_dist_sim_jcn_log'

    solution_df.columns
    gold_solution = solution_df['gold_standard'].values
    points = pd.DataFrame()
    for p in all_paths:
        # p = all_paths[0]
        l, steps = path_length(G, node_ids=p, measure=params['measure'])
        acc = len([v for v in p if v in gold_solution])/len(p)
        points = points.append({'l': l, 'acc': acc, 'p': p}, ignore_index=True)
    points = points.sort_values('l').reset_index(drop=True)
    try:
        pooled_solution = solution_df['pooled'].values
        pooled_l, pooled_steps = path_length(G, node_ids=pooled_solution, measure=params['measure'])
        pooled_acc = len([v for v in pooled_solution if v in gold_solution])/len(pooled_solution)
        pooled = True
    gold_pos = points.loc[points['acc'] == 1].index[0]
    gold_quant = gold_pos/len(points)
    gold_gain = points['x'][gold_pos]/points['x'][0]
    if plot:
        plt.figure()
        plt.scatter(points['l'], points['acc'], color='k', alpha=1/len(unique_ids))
        if pooled:
            plt.scatter(pooled_l, pooled_acc, color='r', alpha=1, marker='x')
            plt.legend(['caminhos em ordem frasal', 'pool 1% mais curtos'])
        else:
            plt.legend(['caminhos em ordem frasal'])
        plt.title('Comparação do comprimento com a acurácia dos caminhos')
        plt.ylabel('Acurácia')
        plt.xlabel('Comprimento do caminho')
        # plt.show()
        plt.savefig('./data/plots/scatter/caminhos_'+sent_id+'.png')
        plt.close()
    del(G)
    # return points
    return gold_quant, gold_gain
# measure_sentence(sent_id, model)
#
# sent_id = 'senseval3.d002.s007'
# measure_sentence(sent_id, dijkstra_frasal, gpickle_folder, params={'measure': 'direct_dist_sim_jcn_log'})

def main():
    # gpickle_folder = './data/sample_10/'
    # folder = 'data/public/'
    folder = 'data/sample/'
    measure = 'direct_dist_sim_jcn_log'
    with open(folder + 'all_solutions.pickle', 'rb') as f:
        global_solution_df = pickle.load(f)[measure]
    with open(folder + 'all_graphs.pickle', 'rb') as f:
        graph_dict = pickle.load(f)
    print('quantidade de grafos:', len(graph_dict))
    golds = pd.DataFrame()
    for s in graph_dict:
        # s = 'semeval2013.d003.s009'
        print('processando', s)
        G = graph_dict[s]
        gold_quant, gold_gain = plot_all_paths(s, global_solution_df, G, params={'measure': measure}, plot=True)
        golds = golds.append(pd.DataFrame({'quant': gold_quant, 'gain': gold_gain}, index=[s]))
    print(golds.head(10))
    # g_quant = np.mean(golds['quant'])
    # g_gain = np.mean(golds['gain'])
    print('rank relativo médio:', np.mean(golds['quant']))
    print('diferença comprimento relativa média:', np.mean(golds['gain']))
    plt.hist(golds['quant'])
    plt.show()
    with open(folder + 'golds.pickle', 'wb') as f:
        pickle.dump(golds, f)


# print(s['degree'] == s['degree_dist'])

if __name__ == "__main__":
    main()
#
# with open('data/results/sample.pickle', 'rb') as f:
#     df = pickle.load(f)
# df.sort_values('clusters', ascending=False)

#
# with open('data/results/sample_solutions.pickle', 'rb') as f:
#     df = pickle.load(f)
# # df.loc['senseval2.d001.s050']
# G
# dicio = {k: v for k, v in G.nodes(data='id')}
# df = pd.DataFrame({'cluster':dicio})
# df['n'] = [int(v.split('.')[-1]) for v in list(df.index)]
# df['k'] = [int(v.split('.')[-2]) for v in list(df.index)]
