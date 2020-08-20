# WSD baseada em Grafos

Reproduzindo o principal resultado de [Sinha 2007], estou montando códigos para realizar desambiguação lexical (WSD) usando de grafos de similaridade entre sentidos de palavras.

# WordNet 3.0 e 1.7

Inicializei trabalhando com a WordNet 3.0, que está diretamente disponível no NLTK.
Entretanto, os dados do SENSEVAL disponíveis no próprio NLTK não estão no padrão dessa WordNet.
O motivo histórico para isso é que o SENSEVAL foi realizado há muito tempo, usando a versão vigente à época.
No caso, essa é a versão 1.7 para a WordNet. Para o SENSEVAL-3 até se utilizou outro repositório de verbos.
Assim, para permitir a reprodução dos resultados de Sinha, estou também trabalhando com a versão 1.7.
Para verificar o trabalho desenvolvido nesse sentido, basta checar o branch _wn17_.

# TSPACO

Estou nesse branch seguindo o trabalho realizado em [Nguyen, 2012], para realizar a WSD usando de ACO (_Ant Colony Optimization_) para resolver um TSP (_Traveling Salesperson Problem_) relacionado à centralidade no grafo de sentidos possíveis das palavras de uma frase. Esse modelo está em processo de desenvolvimento, especialmente na identificação de formas de melhorá-lo.

# Dados
Os dados disponibilizados e utilizados para esse estudo são o _framework_ unificado criado por [Raganato 2017], utilizado largamente em publicações e estudos nessa área.


# Referências
* [Sinha 2007] SINHA, Ravi; MIHALCEA, Rada. __Unsupervised graph-basedword sense disambiguation using measures of word semantic similarity__. In: International conference on semantic computing (ICSC 2007). IEEE, 2007. p. 363-369.
* [Raganato 2017] Raganato, Alessandro & Camacho-Collados, José & Navigli, Roberto. (2017). __Word Sense Disambiguation: A Unified Evaluation Framework and Empirical Comparison__. DOI: 10.18653/v1/E17-1010.
