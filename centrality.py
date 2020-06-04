# -*- coding: utf-8 -*-
# Script para isolar as medidas de similaridade
import networkx as nx

def _example_graph():
    G = nx.Graph()
    G.add_edge(1, 2, weight=1)
    G.add_edge(1, 3, weight=2)
    G.add_edge(2, 3, weight=3)
    return G

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
    # não pode ser usada direto, pq temos similaridade, e tem q ser levada a distancia
    # assim, antes temos que transformar simi(u,v) para dist(u,v)
    H = nx.Graph()
    for u, v, w in G.edges.data('weight'):
        H.add_edge(u, v, weight=1/w)
    # H.edges.data('weight')
    # não tenho certeza se foi assim que a similaridade foi transformada em distância
    return nx.closeness_centrality(H, distance='weight')

def betweenness(G):
    # não pode ser usada direto, pq temos similaridade, e tem q ser levada a distancia
    # assim, antes temos que transformar simi(u,v) para dist(u,v)
    H = nx.Graph()
    for u, v, w in G.edges.data('weight'):
        H.add_edge(u, v, weight=1/w)
    # H.edges.data('weight')
    # não tenho certeza se foi assim que a similaridade foi transformada em distância
    return nx.betweenness_centrality(H, weight='weight')

def pagerank(G):
    return nx.pagerank(G)
