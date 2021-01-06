# -*- coding: utf-8 -*-
# Script para isolar as medidas de similaridade
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
from nltk import FreqDist
# import matplotlib.pyplot as plt

# draw = False

def _example_graph():
    G = nx.read_gpickle('data/sample_total/senseval3.d002.s007.gpickle')
    params = {'measure': 'direct_dist_sim_jcn_log'}
    # print(len(G))
    # nx.draw(G)
    len(G.edges())
    # np.mean([w for (u, v, w) in G.edges(data='dist_sim_jcn_log')])
    # np.mean([w for (u, v, w) in G.edges(data='sim_jcn_log')])
    return G

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
        resp.append(FreqDist(local).max()) # aqui aplica a moda
    return resp

def pooled(G, params=None, draw=False):
    if 'pool_size' in params:
        pool_size = params['pool_size']
    else:
        pool_size = .01
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

    points = pd.DataFrame()
    for p in all_paths:
        # p = all_paths[0]
        l, steps = path_length(G, node_ids=p, measure=params['measure'])
        # acc = len([v for v in p if v in gold_solution])/len(p)
        points = points.append({'l': l, 'p': p}, ignore_index=True)
    points = points.sort_values('l').reset_index(drop=True)
    points = points.head(ceil(len(points) * pool_size))
    pooled_solution = mode_shortest(points['p'])
    return pooled_solution
