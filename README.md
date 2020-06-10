# WSD baseada em Grafos

Reproduzindo o principal resultado de [Sinha 2007], estou montando códigos para realizar desambiguação lexical (WSD) usando de grafos de similaridade entre sentidos de palavras.

# WordNet 3.0 e 1.7

Inicializei trabalhando com a WordNet 3.0, que está diretamente disponível no NLTK.
Entretanto, os dados do SENSEVAL disponíveis no próprio NLTK não estão no padrão dessa WordNet.
O motivo histórico para isso é que o SENSEVAL foi realizado há muito tempo, usando a versão vigente à época.
No caso, essa é a versão 1.7 para a WordNet. Para o SENSEVAL-3 até se utilizou outro repositório de verbos.
Assim, para permitir a reprodução dos resultados de Sinha, estou também trabalhando com a versão 1.7.
Para verificar o trabalho desenvolvido nesse sentido, basta checar o branch _wn17_.
