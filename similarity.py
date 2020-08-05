# -*- coding: utf-8 -*-
# Script para isolar as medidas de similaridade
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from nltk.corpus import stopwords
brown_ic = wordnet_ic.ic('ic-brown.dat')
# semcor_ic = wordnet_ic.ic('ic-semcor.dat')

stopwords.words
# x = wn.synsets('cat', wn.NOUN)[0]
# y = wn.synsets('dog', wn.NOUN)[0]

def lesk(x, y, G=None):
    if G is None:
        return 0
    x_text = [w for w in G.nodes()[x]['gloss'].split()
              if w not in stopwords.words('english')]
    return len(set(x_text) & set(G.nodes()[y].split()))

def ext_lesk(x, y, G):
    x_text = [w for w in G.nodes()[x]['ext_gloss'].split()
              if w not in stopwords.words('english')]
    y_text = [w for w in G.nodes()[y]['ext_gloss'].split()
              if w not in stopwords.words('english')]
    return len(set(x_text) & set(y_text))
