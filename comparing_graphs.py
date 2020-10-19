#!/usr/bin/python
# -*- coding: latin-1 -*-

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


with open('data/results/al_saiagh.pickle', 'rb') as f:
    df = pickle.load(f)
df['nodes'].value_counts()

media = np.mean(df['nodes'])
q1, q2, q3, q4, q5 = np.quantile(df['nodes'], [0.05, 0.25, 0.5, 0.75, 0.95])
l = plt.hist(df['nodes'])
plt.vlines([q1, q2, q3, q4, q5], 0, max(l[0]), colors='r')
plt.vlines([media], 0, max(l[0]), colors='k')
plt.suptitle('Histograma da quantidade de nós')
plt.title('preto: média; vermelho: quantis 5%, 25%, 50%, 75%, 95%', size=9)
# plt.show()
plt.savefig('data/plots/nodes.png')

media = np.mean(df['edges'])
q1, q2, q3, q4, q5 = np.quantile(df['edges'], [0.05, 0.25, 0.5, 0.75, 0.95])
l = plt.hist(df['edges'])
plt.vlines([q1, q2, q3, q4, q5], 0, max(l[0]), colors='r')
plt.vlines([media], 0, max(l[0]), colors='k')
plt.suptitle('Histograma da quantidade de arestas')
plt.title('preto: média; vermelho: quantis 5%, 25%, 50%, 75%, 95%',
          size=9)
# plt.show()
plt.savefig('data/plots/edges.png')
