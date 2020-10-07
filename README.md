# WSD baseada em Grafos

Seguindo na linha geral de [Sinha 2007], estou montando códigos para realizar desambiguação lexical (WSD) usando de grafos de similaridade entre sentidos de palavras. Na atualização da evolução desse projeto, estou me embrenhando mais nas formulações GTSP do problema, de forma a sentir a necessidade de transformar esses grafos em grafos de distância ao invés de similaridade, mais amigáveis a essa outra formulação. Além disso, estou indo mais a fundo em modelagens baseadas em população, começando por GA e passando por ACO, PSO e similares.

# Dados

Os trabalhos modernos utilizam todos o _framework_ unificado criado por [Raganato 2017], linha que também seguimos nesse. Esses dados unificam 5 edições do SemEval/Senseval de tarefas dedicadas a WSD, numa forma de criar um _gold-standard_ para a comparação de tantos modelos que surgiram na área.

Pelo pivotamento para a modelagem baseada em [Fischetti 1997], vou trabalhar todos os dados para que funcionem como GTSP, assim as similaridades devem ser todas transformadas em distâncias. Isso também trará o cuidado de ter grafos realmente *quase-completos* no sentindo de serem completos entre *clusters* mas não entre as cidades de cada *cluster*.

# Modelos Utilizados

## Fischetti - Dijkstra

Uma solução de GTSP apresentada em [Fischetti 1997] (entre outras) consiste em, a partir de uma sequência entre os _clusters_, usar uma variação do algoritmo de _Dijkstra_ para encontrar o melhor caminho entre esses _clusters__. Essa solução é usada em trabalhos posteriores como parte da construção, por exemplo em [Pop 2010], que a usa para solucionar a seleção de _cidades_ dentro de clusters, e as sequẽncias de _clusters_ competem num esquema de GA.

## TSPACO

Estou nesse branch seguindo o trabalho realizado em [Nguyen, 2012], para realizar a WSD usando de ACO (_Ant Colony Optimization_) para resolver um TSP (_Traveling Salesperson Problem_) relacionado à centralidade no grafo de sentidos possíveis das palavras de uma frase. Esse modelo está em processo de desenvolvimento, especialmente na identificação de formas de melhorá-lo.


# Referências
* [Sinha 2007] SINHA, Ravi; MIHALCEA, Rada. __Unsupervised graph-basedword sense disambiguation using measures of word semantic similarity__. In: International conference on semantic computing (ICSC 2007). IEEE, 2007. p. 363-369.
* [Raganato 2017] Raganato, Alessandro & Camacho-Collados, José & Navigli, Roberto. (2017). __Word Sense Disambiguation: A Unified Evaluation Framework and Empirical Comparison__. DOI: 10.18653/v1/E17-1010.
* [Fischetti 1997] Fischetti, Matteo, Juan José Salazar González, and Paolo Toth. "A branch-and-cut algorithm for the symmetric generalized traveling salesman problem." Operations Research 45.3 (1997): 378-394.
* [Pop 2010] Pop, Petrica C., Oliviu Matei, and Cosmin Sabo. "A new approach for solving the generalized traveling salesman problem." International Workshop on Hybrid Metaheuristics. Springer, Berlin, Heidelberg, 2010.
