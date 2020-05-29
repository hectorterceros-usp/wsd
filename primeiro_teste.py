# aqui vou aprender a puxar a WN e usar seus recursos
import nltk
nltk.download() # necessário uma vez só
from nltk.corpus import wordnet as wn


wn.synsets('car')
car_1 = wn.synsets('car')[0]
car_1.__dir__()
car_1.max_depth()
car_1.definition()

dog_1 = wn.synsets('dog')[0]
lch_1 = car_1.lch_similarity(dog_1)
# jcn_1 = car_1.jcn_similarity(dog_1, ) # não funciona sem um corpus associado
path_1 = car_1.path_similarity(dog_1)
print('similaridades: lch: {}; path: {}'.format(lch_1, path_1))
