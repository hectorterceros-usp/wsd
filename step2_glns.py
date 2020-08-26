# -*- coding: utf-8 -*-
# Script para isolar as medidas de similaridade
import networkx as nx
import numpy as np
from step2_heuristic import mfs

import re
# import gzip
import subprocess
# from collections import defaultdict

algo_loc = '../wsd_papers/glns/GLNS.jl/GLNScmd.jl'

def _example_graph():
    # file = 'semeval2013.d009.s008.gpickle'
    sent_id = 'semeval2013.d009.s008'
    G = nx.read_gpickle('data/gpickle/' + sent_id + '.gpickle')
    # print(len(G))
    return G


## Functions

def matrix_from_graph(G):
    l = list(G)
    n = len(l)
    M = np.ones((n, n)) * 9999
    try:
        max_w = max([w for (u, v, w) in G.edges.data('weight')])
    except:
        print('não há edges na frase {}'.format(list(G)[0][:-5]))
        return M
    for (u, v, w) in G.edges.data('weight'):
        l_u, l_v = l.index(u), l.index(v)
        if w < 1e-5:
            w = 1e-5
        M[l_u][l_v] = max_w/w
        M[l_v][l_u] = max_w/w
    # print(np.mean(M))
    return M

def write_matrix(G, sent_id='example'):
    # sent_id = 'senseval2.d000.s032'
    filename = './data/gtsplib/' + sent_id + '.gtsp'
    if len(G) <= 1:
        return ''
    synsets = {}
    for (v, w) in G.nodes(data='id'):
        if w not in synsets:
            synsets[w] = [v]
        else:
            synsets[w].append(v)
    synsets
    M = matrix_from_graph(G)
    with open(filename, 'w') as f:
        f.write('NAME: ' + sent_id)
        f.write('\nTYPE: GTSP')
        f.write('\nCOMMENT: ')
        f.write('\nDIMENSION: ' + str(len(M)))
        f.write('\nGTSP_SETS: ' + str(len(synsets)))
        f.write('\nEDGE_WEIGHT_TYPE: EXPLICIT')
        f.write('\nEDGE_WEIGHT_FORMAT: FULL_MATRIX')
        f.write('\nEDGE_WEIGHT_SECTION')
        # text é uma expressão da matriz de pesos
        text = ''
        for line in M.astype(int).astype(str):
            text += '\n' + ' '.join(line)
        f.write(text)
        #
        f.write('\nGTSP_SET_SECTION:')
        gtsp_clusters = [''] * len(synsets)
        for (v, w) in G.nodes(data='id'):
            word_n = int(w[-3:]) # this is the GTSP cluster id
            v_n = list(G).index(v) + 1 # this is the GTSP vertex id
            # print('word_n = {}, v_n = {}'.format(word_n, v_n))
            gtsp_clusters[word_n] += ' ' + str(v_n)
        for c in range(len(gtsp_clusters)):
            text = '\n' + str(c+1) + ' ' + gtsp_clusters[c] + ' -1'
            # print(text)
            f.write(text)
    return filename

# daqui rodou o GLNS e pegou o resultado
def run_glns_on_matrix(gtsp_loc, G):
    # gtsp_loc = './data/gtsplib/example.gtsp'
    process = subprocess.Popen([algo_loc, gtsp_loc],
                               stdout = subprocess.PIPE,
                               stderr = subprocess.PIPE)
    stdout, stderr = process.communicate()
    # stdout, stderr
    try:
        tour = stdout.split(b'\n')[-3]
    except:
        print('houve algum problema com a instância')
        return []
    tour_vec = eval(re.sub('.*\[', '[', tour.decode()))
    chosen = []
    for i in tour_vec:
        chosen.append(list(G)[i-1])
    chosen
    return chosen

def glns(G):
    filename = write_matrix(G, 'example')
    return run_glns_on_matrix(filename, G)
