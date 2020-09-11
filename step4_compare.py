
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
import matplotlib.pyplot as plt

# para salvar os gtsp preparados
import os
import gzip
from collections import defaultdict
from time import time

# Taking from the step2 models
from step2_heuristic import degree, mfs
from step2_tspaco import single_aco, stochastic_aco, aco
from step2_glns import glns

gpickle_folder = './data/al_saiagh/'

'results.pickle' in os.listdir(gpickle_folder)

resultados = {}
for folder in os.listdir('./data'):
    if '.' in folder:
        continue
    if 'results.pickle' in os.listdir('./data/' + folder):
        with open('./data/' + folder + '/results.pickle', 'rb') as f:
            resultados[folder] = pickle.load(f)

len(resultados)
resultados.keys()

def get_accuracy(resultados, medida='deg_gold'):
    results = {}
    for model_name in resultados:
        # df = resultados['al_saiagh']
        df = resultados[model_name]
        df.columns
        results[model_name] = sum(df[medida])/df.shape[0]
    return results
prec = get_accuracy(resultados, medida='deg_gold')
plt.plot(pd.Series(prec))


def get_precision(resultados, medida='aco_gold'):
    results = {}
    if medida[:3] == 'deg':
        original = 'degree'
    if medida[:3] == 'aco':
        original = 'aco'
    for model_name in resultados:
        # df = resultados['al_saiagh']
        df = resultados[model_name]
        # sum([x is str for x in df[original]])
        try:
            df = df.loc[df[original].astype(str).apply(len) > 2]
        except:
            df = df.loc[df['aco100'].astype(str).apply(len) > 2]
        results[model_name] = sum(df[medida])/df.shape[0]
    return results
prec = get_precision(resultados, medida='aco_gold')
plt.plot(pd.Series(prec))

def compara_modelos(resultados):
    # Agora quero montar a comparação entre várias medidas entre os modelos
    res1 = get_accuracy(resultados, 'deg_gold')
    res2 = get_accuracy(resultados, 'aco_gold')
    plt.plot(pd.Series(res1))
    plt.plot(pd.Series(res2))
    plt.title('Comparação de Acurácias')
    plt.legend(['deg', 'aco'])
    plt.show()
    # Precisões
    res1 = get_precision(resultados, 'deg_gold')
    res2 = get_precision(resultados, 'aco_gold')
    plt.plot(pd.Series(res1))
    plt.plot(pd.Series(res2))
    plt.title('Comparação de Precisões')
    plt.legend(['deg', 'aco'])
    plt.show()
    return None
compara_modelos(resultados)

# está falhando com al_saiagh e jcn+lesk_log
resultados['al_saiagh']
resultados['jcn+lesk']
