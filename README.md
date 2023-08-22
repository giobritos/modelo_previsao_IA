# Projeto de Ciência de Dados com IA - Previsão de Vendas

Este é um projeto de Ciência de Dados que utiliza Inteligência Artificial para prever vendas com base nos gastos em anúncios em três grandes redes: TV, Rádio e Jornal. O projeto foi inspirado no trabalho do Professor João Paulo Rodrigues de Lira da Hashtag Treinamentos.

## Passos do Projeto

1. **Entendimento do Desafio**: O objetivo é prever as vendas com base nos gastos em anúncios nas três redes. Para isso, vamos utilizar um modelo de regressão para fazer as previsões.

2. **Entendimento da Área/Empresa**: A empresa Hashtag investe em anúncios em TV, Rádio e Jornal. Queremos entender como os gastos em cada meio de comunicação impactam as vendas.

3. **Extração/Obtenção de Dados**: Utilizamos uma base de dados chamada "advertising.csv" que contém informações sobre os gastos em anúncios (TV, Rádio, Jornal) e as vendas.

4. **Ajuste de Dados (Tratamento/Limpeza)**: Não foi necessário fazer ajustes nos dados para este projeto, mas em casos reais é comum realizar limpeza e tratamento de dados ausentes ou inconsistentes.

5. **Análise Exploratória**: Fizemos uma análise exploratória dos dados para visualizar a distribuição das informações e a correlação entre os itens.

6. **Modelagem + Algoritmos**: Utilizamos dois modelos de regressão: Regressão Linear e Árvore de Decisão (Inteligência Artificial).

7. **Interpretação de Resultados**: Comparamos os modelos e escolhemos o mais adequado com base no valor de R², que indica o quão bem o modelo explica os dados.

## Instalação das Bibliotecas

Antes de prosseguir, é necessário instalar as bibliotecas necessárias. Execute os seguintes comandos para instalar as bibliotecas matplotlib, seaborn e scikit-learn:

```bash
pip install matplotlib
pip install seaborn
pip install scikit-learn
```

## Importando a Base de Dados

Para iniciar o projeto, importamos a base de dados "advertising.csv" usando a biblioteca Pandas. A tabela contém informações sobre os gastos em TV, Rádio, Jornal e as vendas.

```python
import pandas as pd

tabela = pd.read_csv("advertising.csv")
display(tabela)
```

## Análise Exploratória

Nesta etapa, realizamos uma análise exploratória dos dados. Um dos recursos utilizados é o mapa de calor (heatmap) para visualizar a correlação entre as variáveis.

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(tabela.corr(), annot=True, cmap="Wistia")
plt.show()
```

## Preparação dos Dados e Treinamento do Modelo

A seguir, preparamos os dados para treinamento do modelo de regressão. Separamos os dados em treino e teste, e utilizamos os modelos de Regressão Linear e Árvore de Decisão (Random Forest) para treinamento.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Separando os dados de entrada (X) e saída (y)
y = tabela["Vendas"]
x = tabela.drop("Vendas", axis=1)

# Dividindo os dados em treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=1)

# Criando os modelos de regressão linear e árvore de decisão
modelo_regressao_linear = LinearRegression()
modelo_arvore_decisao = RandomForestRegressor()

# Treinando os modelos
modelo_regressao_linear.fit(x_treino, y_treino)
modelo_arvore_decisao.fit(x_treino, y_treino)
```

## Avaliação dos Modelos

Para avaliar a performance dos modelos, calculamos o valor de R² para cada um deles. Quanto mais próximo de 1, melhor é a explicação do modelo para os dados.

```python
from sklearn import metrics

# Criando as previsões
previsao_regressao_linear = modelo_regressao_linear.predict(x_teste)
previsao_arvore_decisao = modelo_arvore_decisao.predict(x_teste)

# Comparando os modelos com o valor de R²
r2_regressao_linear = metrics.r2_score(y_teste, previsao_regressao_linear)
r2_arvore_decisao = metrics.r2_score(y_teste, previsao_arvore_decisao)

print(f"R² Regressão Linear: {r2_regressao_linear}")
print(f"R² Árvore de Decisão: {r2_arvore_decisao}")
```

## Visualização Gráfica das Previsões

Para visualizar as previsões dos modelos, plotamos um gráfico de linhas com as vendas reais e as previsões feitas pela Árvore de Decisão.

```python
tabela_auxiliar = pd.DataFrame()
tabela_auxiliar["Vendas Reais"] = y_teste
tabela_auxiliar["Previsões Árvore de Decisão"] = previsao_arvore_decisao

plt.figure(figsize=(15, 6))
sns.lineplot(data=tabela_auxiliar)
plt.show()
```

## Nova Previsão

Para fazer uma nova previsão, importamos uma nova tabela com os gastos em TV, Rádio e Jornal, e utilizamos o modelo treinado para fazer a previsão de vendas.

```python
nova_tabela = pd.read_csv("novos.csv")
previsao_nova_tabela = modelo_arvore_decisao.predict(nova_tabela)
print(previsao_nova_tabela)
```

## Importância das Variáveis

Por fim, plotamos um gráfico de barras para visualizar a importância de cada meio de comunicação (TV, Rádio, Jornal) para as vendas.

```python
sns.barplot(x=x_treino.columns, y=modelo_arvore_decisao.feature_importances_)
plt.show()
```

## Executando o Projeto

1. Certifique-se de ter o Python instalado em seu computador.
2. Instale as bibliotecas necessárias com os comandos `pip install matplotlib`, `pip install seaborn` e `pip install scikit-learn`.
3. Faça o download dos arquivos "advertising.csv" e "novos.csv" contendo os dados originais e novos dados para previsão, respectivamente.
4. Execute o código acima para realizar a análise exploratória, treinamento dos modelos, avaliação dos resultados e visualização gráfica.

## Conclusão

Este projeto de Ciência de Dados com Inteligência Artificial foi capaz de prever as vendas com base nos gastos em anúncios nas redes de TV, Rádio e Jornal. O modelo de Árvore de Decisão apresentou melhor desempenho, explicando cerca de 96,25% das variações nas vendas. Além disso, foi possível identificar a importância relativa de cada meio de comunicação para as vendas, com a TV sendo o mais relevante seguido pelo Rádio e Jornal. Com essas informações, a empresa Hashtag pode tomar decisões mais embasadas sobre seus investimentos em publicidade para aumentar as vendas.

## Contribuição

Contribuições são bem-vindas! Se você encontrar problemas ou tiver sugestões para melhorias, sinta-se à vontade para abrir uma issue neste repositório. Caso queira contribuir com código, por favor, abra um pull request para revisão.

## Agradecimentos

Agradeço ao Professor João Paulo Rodrigues de Lira da Hashtag Treinamentos pelo projeto idealizado e à comunidade de código aberto por tornar disponíveis as bibliotecas utilizadas neste projeto. 

**Desenvolvido com :heart: e Python.**
