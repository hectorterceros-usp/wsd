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
from similarity import lesk, ext_lesk
# from centrality import score_vertices
import pickle

pos_dict = {'N': wn.NOUN, 'V': wn.VERB, 'J': wn.ADJ, 'R': wn.ADV}

def clean_context(instance):
    instance_text = instance.context
    instance_pos = instance.word[-1]
    new_context = []
    # checking for 'FRASL', ex. senseval.instances()[21].context
    instance_text = [w for w in instance_text if isinstance(w, tuple)]
    for (w, l) in instance_text:
        # if np.isin(l[0], ['N', 'V', 'J', 'R']):
        if l[0] == str.upper(instance_pos):
            new_label = pos_dict[l[0]]
            new_context.append((w, new_label))
            # print(len(new_context))
    return new_context

def get_synsets(clean_instance, verbose=False):
    synsets = {}
    p = 0 # position
    for (w, l) in clean_instance:
        new_synsets = wn.synsets(w, pos=l)
        if len(new_synsets) < 1 & verbose:
            print('não encontrou synsets para "{}"'.format(w))
        # new_synsets = [{'w': w, 's': s for s in new_synsets]
        # new_synsets = {'word': w, 'synsets': new_synsets}
        synsets[w] = new_synsets
    return synsets

def ext_synset(s, hyper_hypo=True, mero_holo=True, domain=True):
    l = []
    if hyper_hypo:
        # pega os hyper e hypo
        l = l + s.hypernyms()
        l = l + s.hyponyms()
        l = l + s.instance_hypernyms()
        l = l + s.instance_hyponyms()
        l = l + s.verb_groups()
    if mero_holo:
        # pega os mero e holo
        l = l + s.part_meronyms()
        l = l + s.part_holonyms()
        l = l + s.member_meronyms()
        l = l + s.member_holonyms()
        l = l + s.substance_holonyms()
        l = l + s.substance_meronyms()
    if domain:
        # pega os ligados por domínio
        l = l + s.in_usage_domains()
        l = l + s.in_topic_domains()
        l = l + s.in_region_domains()
        l = l + s.topic_domains()
        l = l + s.region_domains()
        l = l + s.usage_domains()
    text = s.definition()
    l = list(set(l))
    for ss in l:
        text += ' . '
        text += ss.definition()
    return text

def get_sim_graph(synsets, dependency=lesk):
    G = nx.Graph()
    l = len(synsets)
    for i in synsets:
        # print(i)
        for t in synsets[i]:
            # populando os vértices
            G.add_node(t.name(), gloss = t.definition(),
                       ext_gloss = ext_synset(t),
                       word = i)
    for i in synsets:
        for j in synsets:
            if j == i:
                continue
            for t in synsets[i]:
                for s in synsets[j]:
                    weight = dependency(t.name(), s.name(), G)
                    G.add_edge(t.name(), s.name(), weight = weight)
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

def check_prediction(result, target_word, original_label, verbose=False):
    try:
        pred = result[target_word]
        correct = [wn.synset(s) for s in SV_SENSE_MAP[original_label]]
        match = pred in correct
        return {'predicted': pred, 'correct': correct, 'match': match}
    except:
        if verbose:
            print('não foi encontrada a target_word: {}'.format(target_word))
        return {'predicted': None, 'correct': None, 'match': 0}

# trabalhando os dados da Senseval-2
def __test__():
    instance = senseval.instances()[10451]
    instance.context
    clean_instance = clean_context(instance)
    synsets = get_synsets(clean_instance)
    G = get_sim_graph(synsets, ext_lesk)
    nx.write_gpickle(G, 'data/tspaco_graph.pkl')
    vertices = score_vertices(G, nx.degree_centrality)
    result = assign_label(clean_instance, synsets, vertices)
    target_word = instance.word[:-2]
    instance_score = check_prediction(result, target_word, instance.senses[0])
