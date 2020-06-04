# -*- coding: utf-8 -*-
# Aqui estou tentando reproduzir o trabalho de Sinha (2007).
# Esse é um trabalho seminal em WSD por grafos.
# Quero construir o processo completo usando das ferramentas disponíveis.
import pandas as pd
from time import time
from similarity import lesk, lch, wup, res, lin, jcn
from centrality import score_vertices, degree, closeness, betweenness, pagerank
from algorithm import *
# clean_context, get_synsets, get_sim_graph, assign_label, check_prediction


# trabalhando os dados da Senseval-2
instance = senseval.instances()[0]
senseval.instances()[0].context
senseval.instances()[0].word
senseval.instances()[0].senses


def score_instance(instance, sim_metric=lesk, cent_metric=degree):
    clean_instance = clean_context(instance)
    synsets = get_synsets(clean_instance)
    G = get_sim_graph(synsets, sim_metric)
    vertices = score_vertices(G, cent_metric)
    result = assign_label(clean_instance, synsets, vertices)

    target_word = instance.word[:-2]
    instance_score = check_prediction(result, target_word, instance.senses[0])
    return instance_score

score_instance(instance)

def score_method(n, sim_metric=lesk, cent_metric=degree):
    scores = pd.DataFrame()
    todos = senseval.instances()
    instances = [i for i in todos if i.word != 'hard-a']
    instances = instances[:n]
    for i in range(len(instances)):
        # i = 10
        score = score_instance(instances[i], sim_metric, cent_metric)
        if score['predicted'] is not None:
            scores = scores.append({'id': i,
                           'real': score['correct'],
                           'pred': score['predicted'],
                           'match': score['match']}, ignore_index=True)
    return scores

def experiment_sim(cent_metric):
    print('---------------------------------')
    print('testando o método {}'.format(cent_metric.__name__))
    sim_list = [lesk, lch, wup, res, lin, jcn]
    best_acc = 0
    best_method = None
    for sim in sim_list:
        start = time()
        scores = score_method(100, sim, cent_metric)
        end = time()
        acc = scores['match'].mean()
        print('método  : {}'.format(sim.__name__))
        print('tempo   : {}'.format(end-start))
        print('acurácia: {}'.format(acc))
        print('---------------------------------')
        if acc > best_acc:
            best_acc = acc
            best_method = sim
    print('melhor acurácia: {} do método {}'.format(best_acc, best_method.__name__))
    return best_method
best_degree = experiment_sim(degree)
best_betweenness = experiment_sim(betweenness)
best_pagerank = experiment_sim(pagerank)
# best_closeness = experiment_sim(closeness)
# Esse ainda não está funcionando, vou seguir trablahando com ele
# cent_metric= closeness
