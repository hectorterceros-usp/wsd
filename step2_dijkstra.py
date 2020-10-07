# -*- coding: utf-8 -*-
# Script para isolar as medidas de similaridade
import networkx as nx
import numpy as np
from collections import defaultdict
import matplotlib
# import matplotlib.pyplot as plt

def _example_graph():
    G = nx.read_gpickle('data/al_saiagh/senseval2.d000.s000.gpickle')
    print(len(G))
    return G

ids = {v: i for v, i in G.nodes(data='id')}
list(set(ids.values()))
cores = Colormap('rainbow', N=len(set(ids.values())))


def dijkstra(LN, inicial = None):
    if inicial == None:
        print('eita')
        return LN
    vs = [v for v, id in LN.nodes(data='id') if id == inicial]
    for v in vs:
        print(nx.dijkstra_path(LN, v, v+'.copy'))
    curtos = {}
    for v in vs:
        curtos[v] = nx.shortest_path_length(LN, v, v+'.copy', weight='dist')
    curtos

def dijkstra_frasal(G):
    # vou começar usando da ordem original da frase
    ids = list(set(dict(G.nodes(data='id')).values()))
    ids.sort(key=lambda x: int(x[-3:]))
    # cmap = matplotlib.cm.get_cmap('rainbow')
    # ainda não consegui colorir o grafo

    LN = nx.DiGraph()
    LN.graph['ids'] = len(ids)

    # LN.add_nodes_from(G)
    # LN.nodes()['semeval2007.d000.s000.t000.c000.copy']
    [LN.add_node(v, id=id) for v, id in G.nodes(data='id')]
    [LN.add_node(v+'.copy', id='final', color=1) for v, id in G.nodes(data='id') if id == inicial]
    node_ids = {v: i for v, i in G.nodes(data='id')}
    id_colors = {'final': 1}
    for i in range(len(ids)):
        id_colors[ids[i]] = i/len(ids)
    node_colors = {k: id_colors[v] for k, v in node_ids.items()}
    nx.set_node_attributes(LN, node_colors, name='color')
    proximo = {}
    inicial = ids[0]
    for i in range(len(ids)):
        if i == len(ids)-1:
            proximo[ids[i]] = 'final'
        else:
            proximo[ids[i]] = ids[i+1]
    LN.nodes()['senseval2.d000.s000.t002.c000']
    LN.nodes(data='color')
    nx.draw(LN,
            node_color=[c for v, c in LN.nodes(data='color')],
            cmap=matplotlib.cm.gist_rainbow)
    for (u, v, w) in nx.DiGraph(G).edges(data='dist'):
        # print(proximo[G.nodes(data='id')[u]])
        if proximo[G.nodes(data='id')[u]] == G.nodes(data='id')[v]:
            LN.add_edge(u, v, dist=w)
        elif proximo[G.nodes(data='id')[u]] == 'final' and G.nodes(data='id')[v] == ids[0]:
            # print('passou aqui')
            LN.add_edge(u, v+'.copy', dist=w)
    len(G.edges())
    len(LN.edges())
    # agora, tenho que rodar o dijkstra nesse DiGraph
    LN = dijkstra(LN, proximo[0])
