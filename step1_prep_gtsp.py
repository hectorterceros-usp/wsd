#!/usr/bin/python

# Esse arquivo é para transformar um problema WSD em GTSP.
# Essa funções servem para qualquer das soluções disponíveis.
# Em seguida, de acordo com a solução escolhida, seguir para seu script.

import xml.etree.ElementTree as ET
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from nltk.corpus import stopwords
brown_ic = wordnet_ic.ic('ic-brown.dat')
from nltk import FreqDist
import numpy as np
import networkx as nx
import pickle
import re
import time
from math import log

# para salvar os gtsp preparados
import os
# import gzip
# from collections import defaultdict
jcn_correction=1

## Functions

def extend_gloss(s, hyper_hypo=True, mero_holo=True, domain=True):
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
    l = list(set(l)) # isso faz com que cada vizinho seja único
    for ss in l:
        text += ' . '
        text += ss.definition()
    text = [w for w in text.split() if w not in stopwords.words('english')]
    # return FreqDist(text)
    return ' . '.join(text)

def lesk_norm(t, s, G=nx.Graph()):
    # LESK feita sobre o G para a aplicação mais simples. Estou testando para ver se funciona do jeito mais simples
    t_def = set(G.nodes(data='ext_gloss')[t].split())
    s_def = set(G.nodes(data='ext_gloss')[s].split())
    intersection = (t_def) & (s_def)
    return len(intersection)

def lesk(t, s, G=nx.Graph()):
    # LESK feita sobre o G para a aplicação mais simples. Estou testando para ver se funciona do jeito mais simples
    t_def = FreqDist(G.nodes(data='ext_gloss')[t].split())
    s_def = FreqDist(G.nodes(data='ext_gloss')[s].split())
    intersection = (t_def) & (s_def)
    value = 0
    for i in intersection:
        value += t_def[i] + s_def[i]
    return value

def lesk_ratio(t, s, G=nx.Graph()):
    # LESK feita sobre o G para a aplicação mais simples. Estou testando para ver se funciona do jeito mais simples
    t_def = FreqDist(G.nodes(data='ext_gloss')[t].split())
    s_def = FreqDist(G.nodes(data='ext_gloss')[s].split())
    intersection = (t_def) & (s_def)
    value = 0
    for i in intersection:
        value += t_def[i] * s_def[i]
    total = sum(t_def.values()) + sum(s_def.values())
    return value/total

def lesk_log(t, s, G=nx.Graph()):
    # LESK feita sobre o G para a aplicação mais simples. Estou testando para ver se funciona do jeito mais simples
    return log(lesk(t, s, G)+1)

def jcn(t, s, G=nx.Graph(), backup=lesk_ratio):
    t_synset = wn.synset(G.nodes(data='synset')[t])
    s_synset = wn.synset(G.nodes(data='synset')[s])
    if (t_synset.pos() == s_synset.pos()) & (t_synset.pos() in ['n', 'v']):
        sim = t_synset.jcn_similarity(s_synset, brown_ic)
        if sim > 1e5: # evitando estourar para os casos sim == 1e300
            sim = 1e5
        # sim_lesk = log(lesk(t, s, G)+1)
        sim_lesk = backup(t, s, G)
        print('JCN: {:.3f}; LSK: {}'.format(sim * jcn_correction, sim_lesk))
        return sim * jcn_correction + sim_lesk
    return backup(t, s, G)

def al_saiagh(t, s, G=nx.Graph(), backup=lesk):
    t_synset = wn.synset(G.nodes(data='synset')[t])
    s_synset = wn.synset(G.nodes(data='synset')[s])
    if (t_synset.pos() == s_synset.pos()) & (t_synset.pos() in ['n', 'v']):
        sim = t_synset.jcn_similarity(s_synset, brown_ic)
        if sim > 1e5: # evitando estourar para os casos sim == 1e300
            sim = 1e5
        sim2 = -1/sim
        # sim_lesk = log(lesk(t, s, G)+1)
        sim_lesk = log(backup(t, s, G)+1)
        print('JCN: {:.3f}; LSK: {}'.format(sim * jcn_correction, sim_lesk))
        return sim2 + sim_lesk
    return log(backup(t, s, G)+1)

def graph_from_synsets(synsets, id, dependency=jcn, backup=lesk):
    # Aqui vou criar tanto o grafo G quanto sua interpretação ed
    # n = sum([len(s) for s in synsets])
    G = nx.Graph()
    for s in synsets:
        # print(len(synsets[s]))
        sense_n = 0  #isso está falhando
        for t in synsets[s]:
            sense_name = s + '.c{:03d}'.format(sense_n)
            G.add_node(sense_name, synset = t.name(), gloss = t.definition(), ext_gloss = extend_gloss(t), id = s)
            sense_n += 1
    for u in list(G):
        for v in list(G):
            if G.nodes()[u]['id'] == G.nodes()[v]['id']:
                continue
            weight = dependency(u, v, G, backup=backup)
            G.add_edge(u, v, weight=weight)
    return G

def write_graph(G, id='example', folder='./data/jcn+lesk_ratio/'):
    filename = folder + id + '.gpickle'
    if len(G) > 1:
        nx.write_gpickle(G, filename)
    return filename

def graph_from_sentence(sent, folder='./data/jcn+lesk_ratio/', dep=(jcn, lesk)):
    id = sent.get('id')
    synsets = {}
    for instance in sent.findall('instance'):
        lemma = instance.get('lemma')
        pos = instance.get('pos')
        synsets_word = wn.synsets(lemma, eval('wn.'+pos))
        synsets[instance.get('id')] = synsets_word

    G = graph_from_synsets(synsets, id, dependency = dep[0], backup=dep[1])
    write_graph(G, id, folder)
    return G



## Datasets

all_data_loc = './data/WSD_Unified_Evaluation_Datasets/ALL/ALL.data.xml'
tree = ET.parse(all_data_loc)
root = tree.getroot()


# Preparing all data
#
# resp = {}
# doc = root[0]
# root.attrib
# sent = root[0][0]
# sent_id = sent.attrib['id']
#
# graph_from_sentence(sent)
#

def run_for_all(folder, dep):
    start = time.time()
    for doc in root:
        for sent in doc:
            # print(sent.get('id'))
            print('processing ' + sent.get('id'))
            graph_from_sentence(sent, folder, dep)
    end = time.time()
    print('demorou {} segundos total'.format(int(end-start)))

# run_for_all()

# Para melhorar esse código,  vou criar os pares folder-function
pares = {'jcn+lesk_ratio': (jcn, lesk_ratio),
         'jcn+lesk_log': (jcn, lesk_log),
         'al_saiagh': (al_saiagh, lesk)}

for dep in pares:
    folder = './data/' + dep + '/'
    try:
        os.listdir(folder)
    except:
        os.mkdir(folder)
    run_for_all(folder, pares[dep])
