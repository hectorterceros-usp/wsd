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

# para salvar os gtsp preparados
import os
import gzip
from collections import defaultdict


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

def lesk(t_ext_gloss='', s_ext_gloss=''):
    # LESK feita sobre o G para a aplicação mais simples. Estou testando para ver se funciona do jeito mais simples
    t_def = set(t_ext_gloss.split())
    s_def = set(t_ext_gloss.split())
    intersection = t_def & s_def
    return len(intersection)

def graph_from_synsets(synsets, id, dependency = 'lesk'):
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
    if dependency == 'lesk':
        for (u, t) in G.nodes(data='ext_gloss'):
            for (v, s) in G.nodes(data='ext_gloss'):
                if G.nodes()[u]['id'] == G.nodes()[v]['id']:
                    continue
                weight = lesk(t, s)
                G.add_edge(u, v, weight=weight)
    return G


def matrix_from_graph(G):
    l = list(G)
    n = len(l)
    M = np.ones((n, n)) * 9999
    for (u, v, w) in G.edges.data('weight'):
        l_u, l_v = l.index(u), l.index(v)
        if w < 1e-5:
            w = 1e-5
        M[l_u][l_v] = 1/w
        M[l_v][l_u] = 1/w
    # print(np.mean(M))
    return M

def write_graph(G, id='example'):
    filename = './data/gpickle/' + id + '.gpickle'
    if len(G) > 1:
        nx.write_gpickle(G, filename)
    return filename

def write_matrix(G, synsets, id='example'):
    filename = './data/gtsplib/' + id + '.gtsp'
    if len(G) < 1:
        return ''
    M = matrix_from_graph(G)

    with open(filename, 'w') as f:
        f.write('NAME: ' + sent.attrib['id'])
        f.write('\nTYPE: GTSP')
        f.write('\nCOMMENT: ')
        f.write('\nDIMENSION: ' + str(len(M)))
        f.write('\nGTSP_SETS: ' + str(len(synsets)))
        f.write('\nEDGE_WEIGHT_TYPE: EXPLICIT')
        f.write('\nEDGE_WEIGHT_FORMAT: FULL_MATRIX')
        f.write('\nEDGE_WEIGHT_SECTION')
        # text é uma expressão da matriz de pesos
        text = ''
        for line in M.astype(int).astype(str):
            text += '\n' + ' '.join(line)
        f.write(text)
        #
        f.write('\nGTSP_SET_SECTION:')
        gtsp_clusters = [''] * len(synsets)
        for (v, w) in G.nodes(data='id'):
            word_n = int(w[-3:]) # this is the GTSP cluster id
            v_n = list(G).index(v) + 1 # this is the GTSP vertex id
            # print('word_n = {}, v_n = {}'.format(word_n, v_n))
            gtsp_clusters[word_n] += ' ' + str(v_n)
        for c in range(len(gtsp_clusters)):
            f.write('\n' + str(c+1) + ' ' + gtsp_clusters[c] + ' -1')
    return filename

def graph_from_sentence(sent, export_graph=True, export_matrix=False):
    id = sent.get('id')
    synsets = {}
    for instance in sent.findall('instance'):
        lemma = instance.get('lemma')
        pos = instance.get('pos')
        synsets_word = wn.synsets(lemma, eval('wn.'+pos))
        synsets[instance.get('id')] = synsets_word

    G = graph_from_synsets(synsets, id)
    if export_graph:
        write_graph(G, id)
    if export_matrix:
        write_matrix(G, synsets, id)
    return G



## Datasets

all_data_loc = './data/WSD_Unified_Evaluation_Datasets/ALL/ALL.data.xml'
tree = ET.parse(all_data_loc)
root = tree.getroot()

all_gold_loc = './data/WSD_Unified_Evaluation_Datasets/ALL/ALL.gold.key.txt'
gold = {}
with open(all_gold_loc, 'r') as f:
    for line in f.readlines():
        gold[line.split()[0]] = line.split()[1:]


# Preparing all data

resp = {}
doc = root[0]
root.attrib
sent = root[0][0]
sent_id = sent.attrib['id']

graph_from_sentence(sent)

import time

start = time.time()
for doc in root:
    for sent in doc:
        # print(sent.get('id'))
        print('processing ' + sent.get('id'))
        graph_from_sentence(sent, export_matrix=True)
end = time.time()
print('demorou {} segundos total'.format(int(end-start)))

#
# 'senseval2.d000.s032.t007'
# sent = root.findall(".//*[@id='senseval2.d000.s032']")[0]
#
# for child in sent:
#     print(child.attrib)
#
# synsets = {}
# for instance in sent.findall('instance'):
#     lemma = instance.get('lemma')
#     pos = instance.get('pos')
#     synsets_word = wn.synsets(lemma, eval('wn.'+pos))
#     synsets[instance.get('id')] = synsets_word
#
# list(G)
# print('processing ' + sent.get('id'))
# G = graph_from_sentence(sent)
# for (v, w) in G.nodes(data='id'):
#     # print(v, w)
#     from_vertex = v[-9:-5]
#     from_id = w[-4:]
#     print(from_vertex, from_id)
