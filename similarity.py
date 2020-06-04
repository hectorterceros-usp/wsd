# -*- coding: utf-8 -*-
# Script para isolar as medidas de similaridade
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
brown_ic = wordnet_ic.ic('ic-brown.dat')
semcor_ic = wordnet_ic.ic('ic-semcor.dat')

# x = wn.synsets('cat', wn.NOUN)[0]
# y = wn.synsets('dog', wn.NOUN)[0]

def lesk(x, y):
    return len(
        set(x.definition().split()) &
        set(y.definition().split()))

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
    return wn.lin_similarity(x, y, brown_ic)

def jcn(x, y):
    # Jiang - Conrath
    return wn.jcn_similarity(x, y, brown_ic)
