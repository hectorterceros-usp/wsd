# -*- coding: utf-8 -*-
# Aqui estou tentando reproduzir o trabalho de Sinha (2007).
# Esse é um trabalho seminal em WSD por grafos.
# Quero construir o processo completo usando das ferramentas disponíveis.
from nltk.corpus import senseval
from nltk.corpus import wordnet as wn
from nltk import FreqDist
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from label_correction import SV_SENSE_MAP
from similarity import lesk
from centrality import score_vertices
from algorithm import *
# clean_context, get_synsets, get_sim_graph, assign_label, check_prediction


# trabalhando os dados da Senseval-2
instance = senseval.instances()[0]
senseval.instances()[0].context
senseval.instances()[0].word
senseval.instances()[0].senses

def score_instance(instance, sim_metric=lesk, cent_metric=nx.degree_centrality):
    clean_instance = clean_context(instance.context)
    synsets = get_synsets(clean_instance)
    G = get_sim_graph(synsets, sim_metric)
    vertices = score_vertices(G, cent_metric)
    result = assign_label(clean_instance, synsets, vertices)

    target_word = instance.word[:-2]
    instance_score = check_prediction(result, target_word, instance.senses[0])
    return instance_score

score_instance(instance)

def score_method(n, sim_metric=lesk, cent_metric=nx.degree_centrality):
    scores = pd.DataFrame()
    instances = senseval.instances()[:n]
    for i in range(len(instances)):
        # i = 10
        score = score_instance(instances[i], sim_metric, cent_metric)
        if score['predicted'] is not None:
            scores = scores.append({'id': i,
                           'real': score['correct'],
                           'pred': score['predicted'],
                           'match': score['match']}, ignore_index=True)
    return scores

scores = score_method(10, lesk, nx.degree_centrality)
scores['match'].mean()
