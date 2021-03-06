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

gpickle_folder = './data/sample/'

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


# measure_sentence(sent_id, model, gpickle_folder)
# measure_sentence('semeval2007.d000.s000', degree_dist, gpickle_folder,params={'measure': 'direct_dist_sim_lesk'})
def plot_sentence(sent_id, global_solution_df, G, params={}):
    # sent_id = 'semeval2013.d006.s024'
    # G = graph_dict[sent_id]
    # global_solution_df = solution_df
    # params={'measure': 'direct_dist_sim_lesk'}
    solution_df = global_solution_df.loc[global_solution_df ['id'] == sent_id]
    n = len(G)
    ids = {k: v for k, v in G.nodes(data='id')}
    # params['measure'] = 'direct_dist_sim_jcn_log'
    pred = solution_df['dijkstra_frasal'].values
    gold_solution = solution_df['gold_standard'].values
    l, steps = path_length(G, node_ids=pred, measure=params['measure'])
    gl, gsteps = path_length(G, node_ids=gold_solution, measure=params['measure'])
    nodes = pd.DataFrame({'name': G.nodes()})
    nodes['c_y'] = nodes['name'].apply(lambda x: int(x[-3:]))
    nodes['t_x'] = nodes['name'].apply(lambda x: int(x[-8:-5]))
    plt.figure()
    plt.scatter(nodes['t_x'], nodes['c_y'], color='k')
    solution_graph = nodes.loc[nodes['name'].isin(pred)].sort_values('t_x')
    solution_y = solution_graph['c_y'].values
    plt.plot(solution_graph['t_x'], solution_graph['c_y'], color='r')
    gold_graph = nodes.loc[nodes['name'].isin(gold_solution)].sort_values('t_x')
    gold_y = gold_graph['c_y'].values
    plt.plot(gold_graph['t_x'], gold_graph['c_y'], color='g')
    for i in range(len(steps)):
        x = i + 0.5
        v = steps[i]
        gv = gsteps[i]
        y = (solution_y[i] + solution_y[i+1])/2
        gy = (gold_y[i] + gold_y[i+1])/2

        plt.annotate('{:.3f}'.format(v), # this is the text
                     (x,y), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(0,10), # distance from text to points (x,y)
                     color = 'r',
                     ha='center') # horizontal alignment ca

        plt.annotate('{:.3f}'.format(gv), # text
                     (x,gy), # point to label
                     textcoords="offset points", # how to position the text
                     xytext=(0,10), # distance from text to points (x,y)
                     color = 'g',
                     ha='center') # horizontal alignment ca
    plt.legend(['dijkstra_frasal, length: {:.2f}'.format(l),
                'gold_standard, length: {:.2f}'.format(gl),
                'conceitos'])
    plt.title('Ilustração dos caminhos')
    plt.ylabel('Conceitos de cada palavra')
    plt.xlabel('Palavras da frase')
    # plt.show()
    plt.savefig('./data/plots/comparacoes/'+sent_id+'.png')
    plt.close()
    del(G)
    return pred
# measure_sentence(sent_id, model)
#
# sent_id = 'senseval3.d002.s007'
# measure_sentence(sent_id, dijkstra_frasal, gpickle_folder, params={'measure': 'direct_dist_sim_jcn_log'})

def main():
    # gpickle_folder = './data/sample_10/'
    with open('./data/results/all_solutions.pickle', 'rb') as f:
        solution_df = pickle.load(f)
    with open('./data/sample/all_graphs.pickle', 'rb') as f:
        graph_dict = pickle.load(f)
    for s in graph_dict:
        print('processando', s)
        G = graph_dict[s]
        f = plot_sentence(s, solution_df, G, params={'measure': 'direct_dist_sim_jcn_log'})


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
