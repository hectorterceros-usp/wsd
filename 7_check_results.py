#!/usr/bin/python

# Esse arquivo é para comparar, no sample que escolhi, os modelos e métricas.
# Em especial, checar se a solução correta tem menor distância.
# Caso contrário, teremos problemas em melhorar os métodos.

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
import pandas as pd
import sys
import pickle

# para salvar os gtsp preparados
import os
import gzip
from collections import defaultdict
from time import time

# Taking from the step2 models
from step2_heuristic import degree, mfs, gold_standard
from step2_tspaco import single_aco, stochastic_aco, aco
from step2_glns import glns
from step2_dijkstra import dijkstra_frasal, dijkstra_pop2010

gpickle_folder = './data/sample/'

all_gold_loc = './data/WSD_Unified_Evaluation_Datasets/ALL/ALL.gold.key.txt'
gold = {}
with open(all_gold_loc, 'r') as f:
    for line in f.readlines():
        gold[line.split()[0]] = line.split()[1:]
gold

# G = nx.read_gpickle('data/sample/semeval2013.d012.s010.gpickle')
# G.edges()[('semeval2013.d012.s010.t004.c007', 'semeval2013.d012.s010.t000.c001')]

with open('data/results/preds.pickle', 'rb') as f:
    s = pickle.load(f)
s
sent_list = list(set([i.split('__')[1] for i in list(s.keys())]))
for sent in sent_list:
    print('###')
    teste = {}
    for model in ['degree', 'dijkstra_frasal', 'dijkstra_pop2010']:
        # print(s[model + '__' + sent])
        teste[model] = s[model + '__' + sent]
    for k, v in teste.items():
        choices = [w.split('.')[3:] for w in v]
        choices.sort()
        # print(choices)
        teste[k] = choices
    expected_value = next(iter(teste.values()))
    all_equal = all(value == expected_value for value in teste.values())
    # print(sent + ': ' + str(all_equal))
    print(sent + ', degree + frasal: ' + str(teste['degree'] == teste['dijkstra_frasal']))
    print(sent + ', degree + pop2010: ' + str(teste['degree'] == teste['dijkstra_pop2010']))
    print(sent + ', pop2010 + frasal: ' + str(teste['dijkstra_pop2010'] == teste['dijkstra_frasal']))

for sent in sent_list:
    print('###')
    teste = {}
    for model in ['degree', 'degree_dist']:
        # print(s[model + '__' + sent])
        teste[model] = s[model + '__' + sent]
    for k, v in teste.items():
        choices = [w.split('.')[3:] for w in v]
        choices.sort()
        # print(choices)
        teste[k] = choices
    expected_value = next(iter(teste.values()))
    all_equal = all(value == expected_value for value in teste.values())
    print(sent + ': ' + str(all_equal))

sent = 'semeval2015.d001.s049'
for model in ['degree', 'dijkstra_frasal', 'dijkstra_pop2010']:
    # print(s[model + '__' + sent])
    # teste[model] = s[model + '__' + 'semeval2015.d001.s049']
    choices = [w.split('.')[3:] for w in s[model + '__' + sent]]
    choices.sort()
    print(choices)

# Checando qual é esse caso em que há pequena discordância
all_data_loc = './data/WSD_Unified_Evaluation_Datasets/ALL/ALL.data.xml'
tree = ET.parse(all_data_loc)
root = tree.getroot()

caso = root.find(".//sentence[@id='semeval2015.d001.s049']")
for w in caso:
    print(w.attrib)
root[0][0]
chutes = wn.synsets('typical')[:2]
for c in wn.synsets('typical'):
    print(c.definition())
    print([l.key() for l in c.lemmas()])
gold['semeval2015.d001.s049.t002']

# Passando a analisar a aplicabilidade da métrica de similaridade
