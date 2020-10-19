# -*- coding: utf-8 -*-
# Script para isolar as medidas de similaridade
import networkx as nx
import numpy as np
from collections import defaultdict

def _example_graph():
    G = nx.read_gpickle('data/jcn+lesk_log/semeval2007.d000.s000.gpickle')
    print(len(G))
    return G

def degree(G, params={}):
    best = {}
    vertices = defaultdict(int)
    for (u, v, w) in G.edges(data='sim'):
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
