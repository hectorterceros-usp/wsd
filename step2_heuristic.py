# -*- coding: utf-8 -*-
# Script para isolar as medidas de similaridade
import networkx as nx
import numpy as np
from collections import defaultdict
import xml.etree.ElementTree as ET # for gold
from nltk.corpus import wordnet as wn # for gold


def _example_graph():
    G = nx.read_gpickle('data/jcn+lesk_log/semeval2007.d000.s000.gpickle')
    print(len(G))
    return G

def degree(G, params={}):
    if 'measure' in params:
        measure = params['measure']
    else:
        measure = 'dist'
    best = {}
    vertices = defaultdict(int)
    for (u, v, w) in G.edges(data=measure):
        if w > 0:
            vertices[v] += w
            vertices[u] += w
    for (v, w) in G.nodes(data='id'):
        d = vertices[v]
        if w not in best:
            best[w] = (v, d)
        elif best[w][1] < d:
            best[w] = (v, d)
    return [v for (v, d) in best.values()]

def mfs(G, params={}):
    best = {}
    for (v, w) in G.nodes(data='id'):
        if v[-3:] == '000':
            best[w] = (v)
    return [best[v] for v in best]

def gold_standard(G, params={}):
    all_gold_loc = './data/WSD_Unified_Evaluation_Datasets/ALL/ALL.gold.key.txt'
    gold = {}
    with open(all_gold_loc, 'r') as f:
        for line in f.readlines():
            gold[line.split()[0]] = line.split()[1:]
    # gold
    best = {}
    for (v, p) in G.nodes(data='id'):
        keys = []
        for l in wn.synset(G.nodes()[v]['synset']).lemmas():
            keys.append(l.key())
        if len(set(gold[p]) & set(keys)) > 0:
            best[p] = v
    return [best[v] for v in best]
