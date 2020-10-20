# -*- coding: utf-8 -*-
# Script para isolar as medidas de similaridade
import networkx as nx
import numpy as np
from collections import defaultdict
import matplotlib
from step2_heuristic import degree
# import matplotlib.pyplot as plt

draw = False

def _example_graph():
    G = nx.read_gpickle('data/sample/semeval2013.d012.s010.gpickle')
    # print(len(G))
    # nx.draw(G)
    len(G.edges())
    return G


def dijkstra(LN, inicial = None, measure='dist'):
    if inicial == None:
        print('eita')
        return [], 0
    vs = [v for v, id in LN.nodes(data='id') if id == inicial]
    melhor_l = np.inf
    for v in vs:
        l = nx.shortest_path_length(LN, v, v+'.copy', weight=measure)
        if l < melhor_l:
            melhor_l = l
            melhor_path = nx.shortest_path(LN, v, v+'.copy', weight=measure)
    return melhor_path[:-1], melhor_l

def create_ln_from_seq(G, ids, measure='dist'):
    # inicial = proximo[0]
    id_colors = {'final': 1}
    for i in range(len(ids)):
        id_colors[ids[i]] = i/len(ids)
    proximo = {}
    inicial = ids[0]
    for i in range(len(ids)):
        if i == len(ids)-1:
            proximo[ids[i]] = 'final'
        else:
            proximo[ids[i]] = ids[i+1]
    LN = nx.DiGraph()
    # LN.graph['ids'] = len(ids)
    node_ids = {v: i for v, i in G.nodes(data='id')}
    node_colors = {k: id_colors[v] for k, v in node_ids.items()}

    # LN.add_nodes_from(G)
    # LN.nodes()['semeval2007.d000.s000.t000.c000.copy']
    [LN.add_node(v, id=id) for v, id in G.nodes(data='id')]
    [LN.add_node(v+'.copy', id='final', color=1) for v, id in G.nodes(data='id') if id == inicial]
    nx.set_node_attributes(LN, node_colors, name='color')
    for (u, v, w) in nx.DiGraph(G).edges(data=measure):
        # Aqui aprender a formar um DiGraph foi importante
        if proximo[G.nodes(data='id')[u]] == G.nodes(data='id')[v]:
            LN.add_edge(u, v, dist=w)
        elif proximo[G.nodes(data='id')[u]] == 'final' and G.nodes(data='id')[v] == ids[0]:
            # print('passou aqui')
            LN.add_edge(u, v+'.copy', dist=w)
    if draw:
        nx.draw(LN,
                node_color=[c for v, c in LN.nodes(data='color')],
                cmap=matplotlib.cm.gist_rainbow)
    return LN

def dijkstra_frasal(G, params=None, draw=False):
    if 'measure' in params:
        measure = params['measure']
    else:
        measure = 'dist'
    # vou começar usando da ordem original da frase
    ids = list(set(dict(G.nodes(data='id')).values()))
    ids.sort(key=lambda x: int(x[-3:]))

    LN = create_ln_from_seq(G, ids, measure)
    # agora, tenho que rodar o dijkstra nesse DiGraph
    try:
        solution, value = dijkstra(LN, ids[0], measure)
    except:
        solution, value = degree(G), 0
    return solution

# Implementando Pop 2010
def fitness(ids, G, measure='dist'):
    LN = create_ln_from_seq(G, ids)
    try:
        solution, value = dijkstra(LN, ids[0], measure)
    except:
        solution, value = degree(G), 0
    return value

def gera_filhos(p1, p2, G, measure = 'dist'):
    # r1 = 0.5
    mutation_prob = 0.05
    genes = len(p1['seq'])
    corte = np.random.randint(genes)
    # inicializando com um parente
    f1 = np.append(p1['seq'][:corte], ['']*(genes-corte))
    f2 = np.append(p2['seq'][:corte], ['']*(genes-corte))
    # completando com o outro parente
    for i in range(corte, genes):
        if p2['seq'][i] not in f1:
            f1[i] = p2['seq'][i]
        if p1['seq'][i] not in f2:
            f2[i] = p1['seq'][i]
    # completando com os clusters que faltam
    # farei de alguma forma direta
    for i in range(corte, genes):
        if f1[i] == '':
            for j in range(corte, genes):
                if p1['seq'][j] not in f1:
                    f1[i] = p1['seq'][j]
        if f2[i] == '':
            for j in range(corte, genes):
                if p2['seq'][j] not in f2:
                    f2[i] = p2['seq'][j]
    # ainda falta aplicar o mutation
    if np.random.rand() <= mutation_prob and genes > 2:
        mutators = np.random.choice(genes, 2, replace=False)
        temp = f1[mutators[0]]
        f1[mutators[0]] = f1[mutators[1]]
        f1[mutators[1]] = temp
    if np.random.rand() <= mutation_prob and genes > 2:
        mutators = np.random.choice(genes, 2, replace=False)
        temp = f2[mutators[0]]
        f2[mutators[0]] = f2[mutators[1]]
        f2[mutators[1]] = temp
    # fitness(f1, G, measure = )
    s1 = {'seq': f1, 'fitness': fitness(f1, G, measure)}
    s2 = {'seq': f2, 'fitness': fitness(f2, G, measure)}
    return s1, s2

# resp = dijkstra_frasal(G)
def pop2010(G, params={'measure': 'sim_als'}):
    if 'measure' in params:
        measure = params['measure']
    else:
        measure = 'dist'
    if len(G.edges) < 1:
        return degree(G)
    n_generations = 5
    # no paper foram entre 500 e 1500... terei que repensar isso
    ids = list(set(dict(G.nodes(data='id')).values()))
    # de acordo com o paper, essa foi a população usada
    pop_size = 2*len(ids)
    ids.sort(key=lambda x: int(x[-3:]))
    # np.random.seed(27)
    pop = []
    # inicializando aleatoriamente, no paper disse q deu na mesma
    for i in range(pop_size):
        ids = np.random.permutation(ids)
        pop.append({'seq': ids, 'fitness': fitness(ids, G, measure)})
    # fazendo a evolução da população
    for t in range(n_generations):
        # ordenando a população por fitness
        pop.sort(key=lambda x: x['fitness'], reverse=True)
        # gerar filhos
        # terei que fazer algum loop para gerar muitos filhos
        # e talvez seja bom já preparar com a fitness correta
        new_pop = []
        while len(new_pop) < pop_size*2:
            p1, p2 = np.random.choice(pop, 2, replace=False)
            f1, f2 = gera_filhos(p1, p2, G, measure)
            new_pop.append(f1)
            new_pop.append(f2)
        pop = pop + new_pop
        # tirar parte da população
        pop.sort(key=lambda x: x['fitness'], reverse=True)
        pop = pop[:pop_size]
        # avaliando a evolução, visualmente
        # print(pop[0]['fitness'])
    chosen = pop[0]
    melhor_path, melhor_l = dijkstra(create_ln_from_seq(G, chosen['seq']), chosen['seq'][0], measure)
    return melhor_path

# pop2010(G, params={'measure': 'dist_als'})
