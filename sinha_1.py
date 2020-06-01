# -*- coding: utf-8 -*-
# Aqui estou tentando reproduzir o trabalho de Sinha (2007).
# Esse é um trabalho seminal em WSD por grafos.
# Quero construir o processo completo usando das ferramentas disponíveis.
from nltk.corpus import senseval
from nltk.corpus import wordnet as wn
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

instance = senseval.instances()[0]

pos_dict = {'N': wn.NOUN, 'V': wn.VERB, 'J': wn.ADJ, 'R': wn.ADV}

def clean_context(instance):
    new_context = []
    for (w, l) in instance.context:
        if np.isin(l[0], ['N', 'V', 'J', 'R']):
            new_label = pos_dict[l[0]]
            new_context.append((w, new_label))
            # print(len(new_context))
    return new_context

clean_instance = clean_context(instance)
# para dar uma limpada nos resultados
clean_instance = clean_instance[:4]

def get_synsets(clean_instance):
    synsets = []
    p = 0 # position
    for (w, l) in clean_instance:
        new_synsets = wn.synsets(w, pos=l)
        if len(new_synsets) < 1:
            print('não encontrou synsets para "{}"'.format(w))
        new_synsets = {'word': w, 'position':p, 'synsets': new_synsets}
        synsets.append(new_synsets)
    return synsets

synsets = get_synsets(clean_instance)
len(instance.context)
len(clean_instance)
[len(s) for s in synsets]
np.prod([len(s) for s in synsets if len(s) > 0])
# só uma palhinha da importância de resolver essa ambiguidade

def get_sim_matrix(synsets, dependency, max_dist = 3):
    G = nx.DiGraph()
    l = len(synsets)
    for i in range(l):
        for j in range(i, l):
            if j - i > max_dist:
                break
            for t in synsets[i]['synsets']:
                for s in synsets[j]['synsets']:
                    weight = dependency(t, s)
                    if weight > 0:
                        G.add_edge(t, s, weight = weight)
    return G

def lesk(x, y):
    return len(
        set(x.definition().split()) &
        set(y.definition().split()))

G = get_sim_matrix(synsets, lesk)
nx.draw(G, with_labels=True)
plt.plot()

len(G)
len(G.edges(data='weight'))
nx.degree_centrality(G)
