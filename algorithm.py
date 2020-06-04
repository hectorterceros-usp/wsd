# -*- coding: utf-8 -*-
# Aqui estou tentando reproduzir o trabalho de Sinha (2007).
# Esse é um trabalho seminal em WSD por grafos.
# Quero construir o processo completo usando das ferramentas disponíveis.
from nltk.corpus import senseval
from nltk.corpus import wordnet as wn
from nltk import FreqDist
from label_correction import SV_SENSE_MAP
import numpy as np
import networkx as nx

pos_dict = {'N': wn.NOUN, 'V': wn.VERB, 'J': wn.ADJ, 'R': wn.ADV}

def clean_context(instance_text):
    new_context = []
    # checking for 'FRASL', ex. senseval.instances()[21].context
    instance_text = [w for w in instance_text if isinstance(w, tuple)]
    for (w, l) in instance_text:
        if np.isin(l[0], ['N', 'V', 'J', 'R']):
            new_label = pos_dict[l[0]]
            new_context.append((w, new_label))
            # print(len(new_context))
    return new_context

def get_synsets(clean_instance, verbose=False):
    synsets = []
    p = 0 # position
    for (w, l) in clean_instance:
        new_synsets = wn.synsets(w, pos=l)
        if len(new_synsets) < 1 & verbose:
            print('não encontrou synsets para "{}"'.format(w))
        # new_synsets = [{'w': w, 's': s for s in new_synsets]
        # new_synsets = {'word': w, 'synsets': new_synsets}
        synsets.append(new_synsets)
    return synsets

def get_sim_graph(synsets, dependency, max_dist = 3):
    G = nx.Graph()
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

def assign_label(clean_instance, synsets, vertices):
    assigned = {}
    for i in range(len(clean_instance)):
        values = [vertices[s] for s in synsets[i]]
        if len(values) < 1:
            chosen = ''
        else:
            chosen = synsets[i][np.argmax(values)]
        assigned[clean_instance[i][0]] = chosen
    return assigned

def check_prediction(result, target_word, original_label):
    try:
        pred = result[target_word]
        correct = [wn.synset(s) for s in SV_SENSE_MAP[original_label]]
        match = pred in correct
        return {'predicted': pred, 'correct': correct, 'match': match}
    except:
        print('não foi encontrada a target_word: {}'.format(target_word))
        return {'predicted': None, 'correct': None, 'match': 0}

# trabalhando os dados da Senseval-2
def __test__():
    instance = senseval.instances()[0]
    clean_instance = clean_context(instance.context)
    synsets = get_synsets(clean_instance)
    G = get_sim_graph(synsets, lesk)
    vertices = score_vertices(G, nx.degree_centrality)
    result = assign_label(clean_instance, synsets, vertices)
    target_word = instance.word[:-2]
    instance_score = check_prediction(result, target_word, instance.senses[0])
