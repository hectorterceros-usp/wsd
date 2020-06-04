# -*- coding: utf-8 -*-
# Script para isolar as medidas de similaridade
import networkx as nx

def score_vertices(G, centrality):
    return centrality(G)
