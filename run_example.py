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
import pandas as pd

# para salvar os gtsp preparados
import os
import gzip
from collections import defaultdict
from time import time

# Taking from the step2 models
from step2_heuristic import degree, mfs
from step2_tspaco import single_aco
from step2_glns import glns


from step1_prep_gtsp import graph_from_sentence

from step3_scoring import score_sentence, score_solution, run_models

# O que quero fazer é preparar um exemplo pequeno que rode facilmente.

# Preparando os grafos
all_data_loc = './data/WSD_Unified_Evaluation_Datasets/ALL/ALL.data.xml'
tree = ET.parse(all_data_loc)
root = tree.getroot()

gpickle_folder='./data/example/'

def run_50(gpickle_folder='./data/example/'):
    try:
        os.listdir(gpickle_folder)
    except:
        os.mkdir(gpickle_folder)
    start = time()
    i = 0
    for doc in root:
        for sent in doc:
            # print(sent.get('id'))
            print('processing ' + sent.get('id'))
            graph_from_sentence(sent, gpickle_folder)
            i += 1
            if i > 50:
                end = time()
                print('demorou {} segundos total'.format(int(end-start)))
                return None


all_gold_loc = './data/WSD_Unified_Evaluation_Datasets/ALL/ALL.gold.key.txt'
gold = {}
with open(all_gold_loc, 'r') as f:
    for line in f.readlines():
        gold[line.split()[0]] = line.split()[1:]
gold



# Functions
# Comparing all models
# results, sol_df = run_models(models=[degree, mfs, aco, glns])
print('###' + gpickle_folder)
# gpickle_folder = './data/' + special_folder + '/'
run_50(gpickle_folder)
results, sol_df = run_models(gpickle_folder, models=['mfs', 'degree', 'single_aco'])
sol_df['gold'] = pd.Series(gold)

def compare_columns(d, c1, c2):
    try:
        return len([v for v in d[c1] if v in d[c2]]) >= 1
    except:
        return False
sol_df['mfs_gold'] = sol_df.apply(compare_columns, axis=1, args=('mfs', 'gold'))
sol_df['aco_gold'] = sol_df.apply(compare_columns, axis=1, args=('aco', 'gold'))
sol_df['deg_gold'] = sol_df.apply(compare_columns, axis=1, args=('degree', 'gold'))
sol_df['aco_mfs'] = sol_df.apply(compare_columns, axis=1, args=('aco', 'mfs'))
sol_df['deg_mfs'] = sol_df.apply(compare_columns, axis=1, args=('deg', 'mfs'))

with open('./data/resultados.pickle', 'wb') as f:
    pickle.dump(sol_df, f)
print('concluído!')
