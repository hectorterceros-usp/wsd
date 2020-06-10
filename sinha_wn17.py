import subprocess
import os
import re
from nltk.corpus import senseval
import networkx as nx
import pickle

def get_definitions(lines):
    senses = []
    synonyms = []
    for l in range(len(lines)):
        # l = 3
        text = lines[l]
        if (re.search('Sense [0-9]+', text) is not None) & (l < len(lines)-2):
            # print(lines[l])
            # print(lines[l+1])
            senses.append(lines[l+1])
            synonyms.append(lines[l+2])
    return senses, synonyms

# get_definitions(lines)

def clean_entry(entry):
    if entry == '':
        return '', ''
    try:
        name, gloss = re.sub('\s* => ', '', entry).split(' -- ')
    except:
        print('{} falhou no clean_entry'.format(entry))
        return '', ''
    return name, gloss

# WN = nx.DiGraph()

def wn_word(word, pos='n', WN=nx.DiGraph(), verbose=False):
    # Carregar "word" na WN (local), sabendo que não já está
    # word, pos = 'interest', 'n'
    resp = subprocess.run('wn {} -hype{} -g'.format(word, pos).split(), stdout=subprocess.PIPE).stdout
    lines = resp.decode('ascii').split('\n')
    l = 6
    n_sense = 0
    # depth=0
    loaded = False # corta a execução antecipadamente se chegar na WN
    while l < len(lines):
        text = lines[l]
        if re.search('Sense [0-9]+', text) is not None:
            n_sense = int(re.search('Sense ([0-9]+)', text).group(1))
            if verbose:
                print('Sense {}'.format(n_sense))
            loaded = False
            # depth = 0
        elif n_sense != 0 and not loaded:
            spaces = len(re.search('^\s*', text).group(0))
            if spaces == 0 and text.strip() != '':
                # print(text)
                # Por algum motivo estava pegando strings vazias
                name, glosa = clean_entry(text)
                if name in WN:
                    # se esse conceito já estiver na WN
                    loaded = True
                WN.add_node(name, gloss=glosa)
                old_name = name
            elif spaces > 0:
                # cur_depth = (spaces - 3)/4
                name, glosa = clean_entry(text)
                if name not in WN:
                    WN.add_edge(old_name, name)
                    old_name = name
                else:
                    # encontrou o resto da WN
                    loaded = True
                #     # Se eu fizesse max_depth diretamente
                # cur_depth = (spaces - 3)/4
                # if cur_depth > depth:
                #     depth = cur_depth
            # print(depth)
        l += 1

def load_corpus_wn(corpus, WN=nx.DiGraph()):
    # presume um corpus de contextos, listas de palavras com POS
    for i in range(len(corpus)):
        # i = corpus[0]
        print('processando frase {} de {}'.format(i, len(corpus)))
        sent = [w for w in corpus[i] if len(w) == 2]
        sent = [(w, 'n') for (w, p) in sent if p[0].lower() == 'n']
        [wn_word(w, p, WN) for (w, p) in sent]
    return WN

#####################################################

senseval.instances()
corpus = [i.context for i in senseval.instances() if i.word != 'hard-a']
len(corpus)

corpus = corpus[:100]
WN = load_corpus_wn(corpus, WN=nx.DiGraph())
len(corpus[2692][0])
corpus[0]

with open('data/wn17.pkl', 'wb') as f:
    pickle.dump(WN, f)


with open('data/wn17.pkl', 'rb') as f:
    WN = pickle.load(f)


WN['interest']
WN['pastime, interest']
WN.nodes['interest, involvement']['gloss']
WN['curiosity, wonder']
WN
#############################3
senseval.instances()[0].context
instance = senseval.instances()[0]
context = instance.context
context = [(w, p[0].lower()) for (w, p) in context if p[0].lower() in ['n', 'v']]
# PATH= $PATH:/usr/local/wordnet1.7/bin

# os.environ['PATH'] = os.environ['PATH']+':/usr/local/wordnet1.7/bin'
teste = subprocess.run('wn dog -synsn'.split())
resp = subprocess.run('wn dog -synsn -g'.split(), stdout=subprocess.PIPE).stdout
resp
lines = resp.decode('ascii').split('\n')

# wn_word('word', 'n', WN)
# len(WN)
# WN = nx.DiGraph()
# WN.nodes
# vou usar max_depth

subprocess.run('wn {} -hype{} -g'.format('word', 'n').split(), stdout=subprocess.PIPE).stdout

cat4 = '''Sense 4
cat-o'-nine-tails, cat -- (a whip with nine knotted cords; "British sailors feared the cat")
       => whip -- (an instrument with a handle and a flexible lash that is used for whipping)
           => instrument -- (a device that requires skill for proper use)
               => device -- (an instrumentality invented for a particular purpose; "the device is small enough to wear on your wrist"; "a device intended to conserve water")
                   => instrumentality, instrumentation -- (an artifact (or system of artifacts) that is instrumental in accomplishing some end)
                       => artifact, artefact -- (a man-made object taken as a whole)
                           => object, physical object -- (a tangible and visible entity; an entity that can cast a shadow; "it was full of rackets, balls and other objects")
                               => entity -- (that which is perceived or known or inferred to have its own physical existence (living or nonliving))
                           => whole, whole thing, unit -- (an assemblage of parts that is regarded as a single entity; "how big is that part compared to the whole?"; "the repairman simply replaced the unit")
                               => entity -- (that which is perceived or known or inferred to have its own physical existence (living or nonliving))
'''

entry = cat4.split('\n')[2]
