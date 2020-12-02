# -*- coding: utf-8 -*-
#!/usr/bin/python

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# vamos melhorar esses gráficos
with open('data/results/sample.pickle', 'rb') as f:
    df = pickle.load(f)
df['nodes'].value_counts()


node0 = df['nodes'].value_counts()[0]
edge0 = df['edges'].value_counts()[0]
df = df.loc[df['edges'] != 0].copy()

# nodes
media = np.mean(df['nodes'])
# q1, q2, q3, q4, q5 = np.quantile(df['nodes'], [0.05, 0.25, 0.5, 0.75, 0.95])
plt.figure()
l = plt.hist(df['nodes'])
plt.xlabel('número de vértices')
plt.ylabel('quantidade na base')
eps = max(l[1])*.03
for q in [0.05, 0.25, 0.5, 0.75, 0.95]:
    n = np.quantile(df['nodes'], q)
    plt.vlines(n, 0, max(l[0]), colors='r')
    # plt.text(n-7, max(l[0])*0.9, '{}%'.format(int(q*100)), rotation=90, color='r')
    plt.text(n-eps, max(l[0])*0.9, '{}% = {}'.format(int(q*100), n), rotation=90, color='r')
plt.vlines([media], 0, max(l[0]), colors='k')
plt.text(media-eps, max(l[0])*0.9, 'média = {}'.format(media), rotation=90, color='k')
plt.suptitle('Histograma da quantidade de nós')
plt.title('total de grafos: {}; zerados: {}'.format(int(sum(l[0])), node0), size=9)
# plt.show()
plt.savefig('data/plots/nodes.png')

# edges
plt.figure()
media = np.mean(df['edges'])
l = plt.hist(df['edges'])
plt.xlabel('número de arestas')
plt.ylabel('quantidade na base')
eps = max(l[1])*.03
for q in [0.05, 0.25, 0.5, 0.75, 0.95]:
    n = np.quantile(df['edges'], q)
    plt.vlines(n, 0, max(l[0]), colors='r')
    # plt.text(n-7, max(l[0])*0.9, '{}%'.format(int(q*100)), rotation=90, color='r')
    plt.text(n-eps, max(l[0])*0.9, '{}% = {}'.format(int(q*100), n), rotation=90, color='r')
plt.vlines([media], 0, max(l[0]), colors='k')
plt.text(media-eps, max(l[0])*0.9, 'média = {}'.format(media), rotation=90, color='k')
plt.suptitle('Histograma da quantidade de arestas')
plt.title('total de grafos: {}; zerados: {}'.format(int(sum(l[0])), edge0), size=9)
# plt.show()
plt.savefig('data/plots/edges.png')


plt.figure()
media = np.mean(df['edges'])
l = plt.hist(df['edges'])
plt.xlabel('número de arestas')
plt.ylabel('quantidade na base')
eps = max(l[1])*.03
for q in [0.05, 0.25, 0.5, 0.75, 0.95]:
    n = np.quantile(df['edges'], q)
    plt.vlines(n, 0, max(l[0]), colors='r')
    # plt.text(n-7, max(l[0])*0.9, '{}%'.format(int(q*100)), rotation=90, color='r')
    plt.text(n-eps, max(l[0])*0.9, '{}% = {}'.format(int(q*100), n), rotation=90, color='r')
plt.vlines([media], 0, max(l[0]), colors='k')
plt.text(media-eps, max(l[0])*0.9, 'média = {}'.format(media), rotation=90, color='k')
plt.suptitle('Histograma da quantidade de arestas')
plt.title('total de grafos: {}; zerados: {}'.format(int(sum(l[0])), edge0), size=9)
# plt.show()
plt.savefig('data/plots/edges.png')
plt.figure()

# clusters
media = np.mean(df['clusters'])
l = plt.hist(df['clusters'])
plt.xlabel('número de clusters')
plt.ylabel('quantidade na base')
eps = max(l[1])*.03
for q in [0.05, 0.25, 0.5, 0.75, 0.95]:
    n = np.quantile(df['clusters'], q)
    plt.vlines(n, 0, max(l[0]), colors='r')
    # plt.text(n-7, max(l[0])*0.9, '{}%'.format(int(q*100)), rotation=90, color='r')
    plt.text(n-eps, max(l[0])*0.9, '{}% = {}'.format(int(q*100), n), rotation=90, color='r')
plt.vlines([media], 0, max(l[0]), colors='k')
plt.text(media-eps, max(l[0])*0.9, 'média = {}'.format(media), rotation=90, color='k')
plt.suptitle('Histograma da quantidade de clusters')
plt.title('total de grafos: {}; zerados: {}'.format(int(sum(l[0])), edge0), size=9)
# plt.show()
plt.savefig('data/plots/clusters.png')
