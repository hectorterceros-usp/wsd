#!/usr/bin/python

# Esse arquivo é para transformar um problema WSD em GTSP.
# Em seguida, aplicar esse problema para o GLNS solver.
# O objetivo com isso é poder comparar esse solver com outros algoritmos.

# Ainda não adaptei para essa pasta, mas vou adaptar tudo nessa pasta
# Para comparar GLNS com outros métodos que ainda vou colocar aqui.
# Em especial, ACO e GA+ACO de [Alsaeedan, 2017]

import xml.etree.ElementTree as ET
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from nltk.corpus import stopwords
brown_ic = wordnet_ic.ic('ic-brown.dat')
from nltk import FreqDist
# from label_correction import SV_SENSE_MAP
import numpy as np
import networkx as nx
# from similarity import lesk, ext_lesk
# from centrality import score_vertices
import pickle
import re

# para rodar o GLNS
import os
import gzip
import subprocess
from collections import defaultdict



all_data_loc = './data/WSD_Unified_Evaluation_Datasets/ALL/ALL.data.xml'
tree = ET.parse(all_data_loc)
root = tree.getroot()
root.attrib
# for child in root:
#     print(child.tag)
len(root)

total_frases = 0
for i in range(len(root)):
    n_frases = len(root[i])
    # print('doc {} tem {} frases'.format(i, n_frases))
    total_frases += n_frases
print('total de frases: {}'.format(total_frases))

all_gold_loc = './data/WSD_Unified_Evaluation_Datasets/ALL/ALL.gold.key.txt'
gold = {}
with open(all_gold_loc, 'r') as f:
    for line in f.readlines():
        gold[line.split()[0]] = line.split()[1:]
gold

# Localização do GLNS em Julia
algo_loc = '../wsd_papers/glns/GLNS.jl/GLNScmd.jl'

# entendendo a estrutura do XML
# <corpus lang="en">
#   <text id="senseval2.d000">
#     <sentence id="senseval2.d000.s000">
#       <wf lemma="the" pos="DET">The</wf>
#       <instance id="senseval2.d000.s000.t000" lemma="art" pos="NOUN">art</instance>


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
    l = list(set(l)) # isso faz com que cada vizinho seja único
    for ss in l:
        text += ' . '
        text += ss.definition()
    text = [w for w in text.split() if w not in stopwords.words('english')]
    return FreqDist(text)

def get_lesk(gloss_t, gloss_s):
    # i, j = 0, 1
    # gloss_t = ed[i]['ed']
    # gloss_s = ed[j]['ed']
    intersection = set(gloss_t.keys()) & set(gloss_s.keys())
    value = 0
    for k in intersection:
        # k = '.'
        value += gloss_t[k] * gloss_s[k]
    return value/(sum(gloss_t.values()) + sum(gloss_s.values()))


def get_sim_graph(synsets):
    # n = sum([len(s) for s in synsets])
    ed = []
    for i in range(len(synsets)):
        # print(i)
        # i = synsets[0]
        for t in synsets[i]:
            # populando os vértices
            # t = synsets[0][0]
            ed.append({'synset': t,
                       'ed': ext_synset(t),
                       'word': i+1})
    return ed

def create_matrix(ed):
    n = len(ed)
    M = np.ones((n, n)) * 9999
    for i in range(n):
        for j in range(i+1, n):
            if ed[i]['word'] == ed[j]['word']:
                continue
            try:
                weight = wn.jcn_similarity(ed[i]['synset'], ed[j]['synset'], brown_ic)
                assert(weight > 0)
                # print('jcn weight: {}'.format(weight))
            except:
                print('começou lesk')
                weight = get_lesk(ed[i]['ed'], ed[i]['ed'])
                print('lsk weight: {}'.format(weight))
            if weight > 1e-5: # evitar coisas como 1e-300 da jcn
                M[i][j] = 1/weight
                M[j][i] = 1/weight
    return M


resp = {}
doc = root[0]
for sent in doc:
    # sent = root[0][0]
    sent_id = sent.attrib['id']

    synsets = []
    for instance in sent.findall('instance'):
        lemma = instance.get('lemma')
        pos = instance.get('pos')
        # print(lemma, pos)
        synsets_word = wn.synsets(lemma, eval('wn.'+pos))
        print(len(synsets_word))
        synsets.append(synsets_word)
    synsets
    if len(synsets) == 0:
        continue

    ed = get_sim_graph(synsets)
    M = create_matrix(ed)
    np.mean(M)

    def gtsp_matrix(M):
        text = ''
        for line in M.astype(int).astype(str):
            text += '\n' + ' '.join(line)
        return text

    sent.attrib
    gtsp_loc = './data/gtsp_clean/' + sent.attrib['id'] + '.gtsp'
    with open(gtsp_loc, 'w') as f:
        f.write('NAME: ' + sent.attrib['id'])
        f.write('\nTYPE: GTSP')
        f.write('\nCOMMENT: ')
        f.write('\nDIMENSION: ' + str(len(M)))
        f.write('\nGTSP_SETS: ' + str(len(synsets)))
        f.write('\nEDGE_WEIGHT_TYPE: EXPLICIT')
        f.write('\nEDGE_WEIGHT_FORMAT: FULL_MATRIX')
        f.write('\nEDGE_WEIGHT_SECTION')
        # aqui coloco a matrix dos pesos
        f.write(gtsp_matrix(M))
        #
        f.write('\nGTSP_SET_SECTION:')
        c = 0
        for w in range(len(synsets)):
            text = str(w+1) + ' '
            for s in range(len(synsets[w])):
                c += 1
                text += str(c) + ' '
            f.write('\n' + text + '-1')

    # daqui rodou o GLNS e pegou o resultado
    # gtsp_loc = './gtsp_clean/senseval2.d000.s000.gtsp'
    process = subprocess.Popen([algo_loc, gtsp_loc],
                               stdout = subprocess.PIPE,
                               stderr = subprocess.PIPE)
    stdout, stderr = process.communicate()
    stdout, stderr
    try:
        tour = stdout.split(b'\n')[-3]
    except:
        continue
    tour_vec = eval(re.sub('.*\[', '[', tour.decode()))
    # Terei que arrumar a criação do GTSP para poder recuperar os Synsets do Tour

    chosen = []
    for i in tour_vec:
        chosen.append(ed[i-1])
    chosen

    correct = 0
    for w in chosen:
        # w = chosen[0]
        id = '{}.t{:03d}'.format(sent_id, w['word']-1)
        if w['synset'].lemmas()[0].key() in gold[id]:
            correct += 1
    print('{} corretos de {}'.format(correct, len(chosen)))
    resp[sent.attrib['id']] = (correct, len(chosen))

corretos = sum([c for (c, t) in resp.values()])
total = sum([t for (c, t) in resp.values()])
print('Total: {} corretos de {}'.format(corretos, total))
