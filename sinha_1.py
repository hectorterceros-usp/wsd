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
# clean_instance = clean_instance[:4]

def get_synsets(clean_instance):
    synsets = []
    p = 0 # position
    for (w, l) in clean_instance:
        new_synsets = wn.synsets(w, pos=l)
        if len(new_synsets) < 1:
            print('não encontrou synsets para "{}"'.format(w))
        # new_synsets = [{'w': w, 's': s for s in new_synsets]
        # new_synsets = {'word': w, 'synsets': new_synsets}
        synsets.append(new_synsets)
    return synsets

synsets = get_synsets(clean_instance)
len(instance.context)
len(clean_instance)
[len(s) for s in synsets]
np.prod([len(s) for s in synsets if len(s) > 0])
# só uma palhinha da importância de resolver essa ambiguidade

def get_sim_graph(synsets, dependency, max_dist = 3):
    G = nx.DiGraph()
    l = len(synsets)
    for i in range(l):
        for j in range(i, l):
            if j - i > max_dist:
                break
            for t in synsets[i]:
                for s in synsets[j]:
                    weight = dependency(t, s)
                    if weight > 0:
                        G
                        G.add_edge(t, s, weight = weight)
    return G

def lesk(x, y):
    return len(
        set(x.definition().split()) &
        set(y.definition().split()))

G = get_sim_graph(synsets, lesk)
nx.draw(G, with_labels=True)
plt.plot()

len(G)
len(G.edges(data='weight'))

def score_vertices(G, centrality):
    return centrality(G)

# vertices = nx.degree_centrality(G)
vertices = score_vertices(G, nx.degree_centrality)

def assign_label(clean_instance, synsets, vertices):
    assigned = []
    for i in range(len(clean_instance)):
        values = [vertices[s] for s in synsets[i]]
        if len(values) < 1:
            chosen = ''
        else:
            chosen = synsets[i][np.argmax(values)]
        assigned.append((clean_instance[i][0], chosen))
    return assigned

result = assign_label(clean_instance, synsets, vertices)
result
