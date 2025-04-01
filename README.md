# Descrição
Este repositório contém todos os note necessários para replicar os experimentos do TCC - Análise da Defesa de Grandes Modelos de Linguagem Com Uso de Marca D'água. Esses arqvuios estão em formato de notebooks para facilitar a compreensão deles. Além disso, eles são divididos em quatro pastas, uma para cada etapa do processo:
- "Dataset Pre-Processing" - Pré-processamento dos datasets
- "Pre-Training" - Pré-treinamento do modelo controle, ou seja, sem a marca d'água
- "Watermarking Pre-Training" - Pré-treinamento marcado, ou seja, pré-treinamento do modelo marcado
- "Model Extraction" - Extração de um modelo pré-treinado

# Observações
Aqui estão algumas observações quanto à execução dos arquivos deste repositório.

Uma observação geral é que a primeira célula de cada notebook define as variáveis gerais que serão usadas em sua execução. Portanto, elas podem ser alteradas de acordo com o que o usuário deseje em seus experimentos.

## Dataset Pre-Processing
Os arquivos da pasta de pré-processamento devem ser executados antes de todos os outros, já que eles são responsáveis por montar os datasets usados nos treinamentos.

Os arquivos dessa pasta possuem os nomes dos datasets que eles geram, sendo que o pré-processamento de alguns datasets tiveram de ser divididos em duas partes. Logo, para o Dataset ser montado, é necessário executar essas partes na ordem correta.

Em todos os casos o arquivo de Parte 1 pega o Dataset bruto original e converte sua informação em uma série de arquivos de texto, os quais são processados pelo respectivo arquivo de Parte 2 para formar o Dataset que usado para o pré-treinamento.

É necessário garantir que a pasta de salvamento dos arquivos de texto gerados na Parte 1 realmente exista, caso contrário o script encontrará um erro de execução. Vale notar, a pasta que está definida por padrão no arquivo, não existe neste repositório, então é necessário cria-la manualmente antes de executar o notebook.

Os arquivos de pré-treinamento dos modelos esperam os datasets já tokenizados, então é necessário executar o notebook "Tokenization of Custom Datasets" antes de realizar o treinamento, mas após os Datasets serem pré-processados.

## Pre-Training & Watermarked_Pretraining
As pastas de pré-treinamento e pré-treinamento marcado contém dois arquivos: um para o respectivo treinamento e outro com o sufixo "Model Accuracy Analysis". Esse segundo arquivo serve para calcular a acurácia do modelo na respectiva tarefa do treinamento.

Em mais detalhes, o "Pre-Trained Model Accuracy Analysis" calcula a acurácia de um modelo somente em relação à suas tarefas de pré-treinamento, e o arquivo "Watermarked Model Accuracy Analysis" também considera a acurácia do arquivo em relação a sua capacidade de previsão da marca d'água.

## Model Extraction
A primeira parte da extração de modelo é realizada pelo arquivo "Extraction Dataset Creation" na pasta "Dataset Pre-Processing". Ele que é responsável por criar o dataset que extrai o conhecimento de um modelo.

Já o arquivo "Model Extraction" da pasta de mesmo nome, é quem usa esse Dataset para treinar um modelo usurpador.

# Ordem de Execução da Experimentação Original
1. Pré-processar os datasets do Bookcorpus e da Wikipedia.
2. Tokenizar esses dois datasets.
3. Realizar o pré-treinamento não marcado de um modelo.
4. Validar a acurácia do modelo não marcado em suas tarefas de objetivo.
5. Treinar o pré-treinamento com marca d'água de um modelo.
6. Validar a acurácia do modelo marcado em suas tarefas de objetivo e na marca d'água.
7. Pré-processar o dataset Open Web Text.
8. Usar o Open Web Text para criar um dataset de extração para os modelos marcado e não marcado.
9. Treinar dois usupadores, um para cada dataset de extração gerado.
10. Validar a acurácia dos dois usurpadores na tarefa de objetivo, e validar a acurácia do usurpador marcado na marca d'água.
