# -*- coding: utf-8 -*-
# Script para isolar as medidas de similaridade
import networkx as nx
import numpy as np
from step2_heuristic import mfs
import random

q0 = 0.9
global_theta = 1
global_beta = 2
global_tau = 1
global_lambda = 0.1
epsilon = 0.1

def _example_graph():
    sent_id = 'senseval3.d002.s007'
    H = nx.read_gpickle('data/sample/' + sent_id + '.gpickle')
    # print(len(H))
    return H
# G = _example_graph()
# list(G.nodes)
# vertex, neighbor = 'senseval3.d002.s007.t000.c005', 'senseval3.d002.s007.t006.c002'
#
# for n in G.nodes():
#     # print(n)
#     G.nodes[n]['p'] = 1
# # G['wrinkle.n.01']['p']
#
# words = []
# for v in G.nodes:
#     w = G.nodes()[v]['id']
#     if w not in words:
#         words.append(w)
# words
#
# for v in G.nodes(data='id'):
#     print(v[1])

class Ant():
    def __init__(self, vertex, G, n_words = None, measure='sim_jcn_log'):
        # self.v = vertex # vou usar o path e olhar para o final
        self.G = G
        self.path = [vertex]
        self.path_words = []
        if n_words == None:
            n_words = len(set([w for (v, w) in G.nodes(data='id')]))
        self.n_words = n_words
        self.path_length = 0
        self.measure = measure
        return None

    def current(self):
        return self.path[-1]

    def move(self, beta = global_beta):
        # vertex = self.v # para permitir teste local
        vertex = self.current()
        self.path_words.append(self.G.nodes()[vertex]['id'])
        # lista de vizinhos e dicionário com seus valores
        l = list(self.G.neighbors(vertex))
        # com default = 0, os clusters já visitados não conseguem ser
        # revisitados, independente do fator aleatório.
        d = [-1]*len(l)
        # computando a memória local, a visão dos vizinhos
        for n in range(len(l)):
            neighbor = l[n]
            if self.G.nodes()[neighbor]['id'] in self.path_words:
                continue
            # seria mais elegante rodar em l, mas eu preciso de d ordenada
            # estou com problema de convergencia, vou forçar
            else:
                p = self.G.nodes[neighbor]['p']
                w = self.G.edges[vertex, neighbor][self.measure]
                d[n] = w * (p ** beta)
        # escolhendo modo de avanço - otimizado ou genérico
        q = np.random.random()
        if q < q0:
            next = np.argmax(d)
        else:
            s = np.random.random(len(l))
            next = np.argmax(s * d)
        self.path.append(l[next])
        self.path_length += d[next]
        # self.v = l[next]
        return l[next]

    def is_route_completed(self):
        completed = len(self.path) >= self.n_words
        if completed:
            # completamos uma rota
            self.global_update()
            # reset
            self.path_words = []
            self.path_length = 0
            self.path = self.path[-1:]
        return completed

    def global_update(self):
        # G.graph['global_best'] = 0
        # global_best = G.graph['global_best']
        better = self.path_length > self.G.graph['best_length']
        if better:
            # print('New best! Length = {}'.format(self.path_length))
            self.G.graph['best_length'] = self.path_length
            self.G.graph['best_path'] = self.path
            for n in self.path:
                self.G.nodes()[n]['p'] += 1/self.path_length
                # global_update, preciso confirmar
        return better

def aco(H, iter=3, theta = 1, lam = 0.5, tau = 1, measure='sim_jcn_log'):
    # Primeiro vou testar se há mais de duas instâncias a desambiguar
    G = H.copy()
    if len(G.edges()) < 1:
        for v in G.nodes():
            G.nodes()[v]['p'] = theta
        return G
    # Preciso corrigir n_words, mas é algo do grafo, não entra como param
    n_words = len(set([w for (v, w) in G.nodes(data='id')]))
    G.graph['best_length'] = 0
    G.graph['best_path'] = []
    ants = []
    for v in G.nodes():
        # vamos localizar uma formiga em cada vértice
        G.nodes()[v]['p'] = theta
        ants.append(Ant(v, G, n_words, measure=measure))
    for i in range(iter):
        # print(ants[0].path)
        for j in range(n_words):
            for ant in ants:
                ant.move()
            for v in G.nodes():
                # lam para lambda, que é palavra reservada em python
                # Aqui temos a evaporação do feromônio
                G.nodes()[v]['p'] = (1 - lam) * G.nodes()[v]['p']
            for ant in ants:
                # Aqui temos a deposição de feromônio pelas formigas
                G.nodes()[ant.current()]['p'] += lam * tau
        for ant in ants:
            # Ao final da iteração, temos a avaliação de cada rota criada
            ant.is_route_completed()
        # ainda não implementei a outra forma parar a otimização
        # que é quando todas as formigas seguem o mesmo caminho
        # mas acho que não precisa por ora
        # pode valer a pena quando eu rodar para todos os grafos
        # terei que pensar em como isso seria. comparando paths?
    # When out from this loop, optimization has stopped
    return G

def single_aco(H, params={}):
    # correção de default por params
    if len(H.edges()) < 1:
        return []
    default = {'iter':5, 'theta': 1, 'lam': 0.5, 'tau': 1, 'measure': 'sim_jcn_log'}
    for key in default:
        if key in params:
            default[key] = params[key]
    # Rodando a otimização
    G = aco(H, default['iter'], default['theta'], default['lam'], default['tau'], default['measure'])
    # Extraindo os melhores valores
    values = dict([(v, p) for (v, p) in G.nodes(data='p')])
    best = {}
    for (v, w) in G.nodes(data='id'):
        d = values[v]
        if w not in best:
            best[w] = (v, d)
        elif best[w][1] < d:
            best[w] = (v, d)
    return [v for (v, d) in best.values()]

def stochastic_aco(H, params={}):
    if len(H.edges()) < 1:
        return []
    best_result = 0
    chosen = H
    n_words = len(set([w for (v, w) in H.nodes(data='id')]))
    default = {'iter':3, 'theta': 1, 'lam': 0.5, 'tau': 1, 'runs':10, 'random_seed':7, 'measure': 'sim_jcn_log'}
    for key in default:
        if key in params:
            default[key] = params[key]
    # random seed
    random.seed(default['random_seed'])
    for run in range(default['runs']):
        G = aco(H, default['iter'], default['theta'], default['lam'], default['tau'], default['measure'])
        if G.graph['best_length'] > best_result:
            chosen = G
    values = dict([(v, p) for (v, p) in chosen.nodes(data='p')])
    best = {}
    # print('stochastic length: ' + str(len(chosen)))
    # print('values: ' + str(len(values)))
    for (v, w) in chosen.nodes(data='id'):
        d = values[v]
        if d is None:
            print("houve um problema para o grafo " + str(w))
            continue
        # print(d, w, v)
        if w not in best:
            best[w] = (v, d)
        elif best[w][1] < d:
            best[w] = (v, d)
    return [v for (v, d) in best.values()]

# r = aco(G, iter=100, lam=0.5)
# G.graph['best_path']
# G.graph['best_length']
# words
#
# G.nodes()['cement.n.03']['id']
#
# def score_vertices(G, centrality):
#     return centrality(G)
