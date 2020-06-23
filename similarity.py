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

def lesk(x, y):
    x_text = [w for w in x.definition().split()
              if w not in stopwords.words('english')]
    return len(set(x_text) & set(y.definition().split()))

def simp_lesk(x, y):
    # Há uma simplificação de Lesk, que ainda não montei.
    # Essa é assimétrica, usando de um contexto mais genérico.
    # Ainda vou pegar para testar, se valer a pena
    return lesk(x, y)

def lch(x, y):
    # Leacock - Chodorow
    return wn.lch_similarity(x, y)

def wup(x, y):
    # Wu - Palmer
    return wn.wup_similarity(x, y)

def res(x, y):
    # Resnik
    return wn.res_similarity(x, y, brown_ic)

def lin(x, y):
    # Lin
    if x == wn.synset('entity.n.01') or y == wn.synset('entity.n.01'):
        return 0
    return wn.lin_similarity(x, y, brown_ic)

def jcn(x, y):
    # Jiang - Conrath
    return wn.jcn_similarity(x, y, brown_ic)
