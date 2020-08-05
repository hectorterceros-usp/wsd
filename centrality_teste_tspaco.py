# -*- coding: utf-8 -*-
# Script para isolar as medidas de similaridade
import networkx as nx
import numpy as np

q0 = 0.9
global_theta = 1
global_beta = 2
global_tau = 1
global_lambda = 0.1
epsilon = 0.1

def _example_graph():
    G = nx.read_gpickle('data/tspaco_graph.pkl')
    print(len(G))
    return G
list(G.nodes)

for n in G.nodes():
    # print(n)
    G.nodes[n]['p'] = 1
G['wrinkle.n.01']['p']
G.nodes()['wrinkle.n.01']['p']
G.nodes['wrinkle.n.01']['p']
l = list(G.neighbors('wrinkle.n.01'))
vertex = 'wrinkle.n.01'
neighbor = 'source.n.07'
G.edges[vertex, neighbor]['weight']

words = []
for v in G.nodes:
    w = G.nodes()[v]['word']
    if w not in words:
        words.append(w)
words

for v in G.nodes(data='word'):
    print(v[1])

class Ant():
    def __init__(self, vertex, G, n_words = None):
        # self.v = vertex # vou usar o path e olhar para o final
        self.G = G
        self.path = [vertex]
        self.path_words = []
        if n_words == None:
            n_words = len(set([w for (v, w) in G.nodes(data='word')]))
        self.n_words = n_words
        self.path_length = 0
        return None

    def current(self):
        return self.path[-1]

    def move(self, beta = global_beta):
        # vertex = self.v # para permitir teste local
        vertex = self.current()
        self.path_words.append(G.nodes()[vertex]['word'])
        # lista de vizinhos e dicionário com seus valores
        l = list(G.neighbors(vertex))
        # com default = 0, os clusters já visitados não conseguem ser
        # revisitados, independente do fator aleatório.
        d = [-1]*len(l)
        # computando a memória local, a visão dos vizinhos
        for n in range(len(l)):
            neighbor = l[n]
            if G.nodes()[neighbor]['word'] in self.path_words:
                continue
            # seria mais elegante rodar em l, mas eu preciso de d ordenada
            # estou com problema de convergencia, vou forçar
            else:
                p = G.nodes[neighbor]['p']
                w = G.edges[vertex, neighbor]['weight']
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
        better = self.path_length > G.graph['best_length']
        if better:
            print('New best! Length = {}'.format(self.path_length))
            G.graph['best_length'] = self.path_length
            G.graph['best_path'] = self.path
            for n in self.path:
                G.nodes()[n]['p'] += 1/self.path_length
                # global_update, preciso confirmar
        return better

def aco(G, iter=10, n_words = None, theta = 1, lam = 0.5, tau = 1):
    G.graph['best_length'] = 0
    G.graph['best_path'] = []
    ants = []
    for v in G.nodes():
        # vamos localizar uma formiga em cada vértice
        ants.append(Ant(v, G, n_words))
        G.nodes()[v]['p'] = theta
    for i in range(iter):
        print(ants[0].path)
        for ant in ants:
            ant.move()
            ant.is_route_completed()
        for v in G.nodes():
            # lam para lambda, que é palavra reservada em python
            G.nodes()[v]['p'] = (1 - lam) * G.nodes()[v]['p']
        for ant in ants:
            G.nodes()[ant.current()]['p'] += lam * tau
            # isso está um pouco feio. Vejo que é para depositar após a
            # evaporação, mas talvez tenha um jeito melhor de fazer
            # pensando bem, não entendi
        # ainda não implementei a outra forma parar a otimização
        # que é quando todas as formigas seguem o mesmo caminho
        # mas acho que não precisa por ora
        # pode valer a pena quando eu rodar para todos os grafos
        # terei que pensar em como isso seria. comparando paths?
    return dict([(v, p) for (v, p) in G.nodes(data='p')])

r = aco(G, iter=100, lam=0.5)
G.graph['best_path']
G.graph['best_length']
words


def score_vertices(G, centrality):
    return centrality(G)

def degree(G):
    # nx.degree_centrality(G) não serve, pois não usa pesos
    for v in G.nodes:
        G.nodes[v]['weight'] = 0
    for u, v, w in G.edges.data('weight'):
        G.nodes[u]['weight'] += w
        G.nodes[v]['weight'] += w
    output = {}
    for v in G.nodes:
        output[v] = G.nodes[v]['weight']
    return output

def closeness(G):
    print('closeness abandonado, respondendo degree')
    return degree(G)

def betweenness(G):
    print('betweenness abandonado, respondendo degree')
    return degree(G)

def pagerank(G):
    return nx.pagerank(G)
