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
from time import time

# Taking from the step2 models
from step2_heuristic import degree_centrality, mfs
from step2_tspaco import aco
from step2_glns import glns

all_gold_loc = './data/WSD_Unified_Evaluation_Datasets/ALL/ALL.gold.key.txt'
gold = {}
with open(all_gold_loc, 'r') as f:
    for line in f.readlines():
        gold[line.split()[0]] = line.split()[1:]
gold


# Functions
def score_sentence(sent_id, model):
    # sent_id = 'senseval2.d001.s003'
    gpickle_file = './data/gpickle/' + sent_id + '.gpickle'
    G = nx.read_gpickle(gpickle_file)
    # list(G)
    pred = model(G)
    tp, fp = 0, 0
    for p in pred:
        # p = pred[0]
        inst_id = p[:-5]
        keys = []
        for l in wn.synset(G.nodes()[p]['synset']).lemmas():
            keys.append(l.key())
        if len(set(gold[inst_id]) & set(keys)) > 0:
            tp += 1
        else:
            fp += 1
    del(G)
    return tp, fp
# score_sentence(sent_id, model)

def score_solution(model, n=-1):
    sent_list = os.listdir('./data/gpickle/')
    sent_result = {}
    sent_list = sent_list[:n]
    for file in sent_list:
        # file = sent_list[0]
        sent_id = file[:-8]
        tp, fp = score_sentence(sent_id, model)
        if tp + fp == 0:
            continue
        acc = tp / (tp + fp)
        sent_result[sent_id] = {'tp': tp, 'fp': fp, 'acc': acc}
    sum_tp = sum([sent['tp'] for sent in sent_result.values()])
    sum_fp = sum([sent['fp'] for sent in sent_result.values()])
    avg_acc = np.mean([sent['acc'] for sent in sent_result.values()])
    sum_acc = sum_tp / (sum_tp + sum_fp)
    print(sum_tp, sum_fp, avg_acc, sum_acc)
    return (sum_tp, sum_fp, avg_acc, sum_acc)


# Comparing models
def run_models(n=-1):
    results = {}
    for model in [degree_centrality, mfs, aco, glns]:
        start = time()
        r = score_solution(model, n)
        results[model.__name__] = r
        end = time()
        print('demorou {} segundos total'.format(int(end-start)))
    return results


# Comparing all models
run_models(20)
