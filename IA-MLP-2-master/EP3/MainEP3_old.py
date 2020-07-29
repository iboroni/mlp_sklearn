"""
    Classe responsavel por por controlar toda a execução do algoritmo Multi-layer Perceptron, tanto para sua fase
    treinamento, quanto para sua fase de teste
"""
from sklearn.neural_network import MLPClassifier  # implementa o MLP
from sklearn.model_selection import train_test_split  # define os conjuntos de treino e teste
from sklearn.preprocessing import StandardScaler  # normalizacao
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # metricas
from sklearn.model_selection import GridSearchCV  # pre-fit. otimizador dos hyperparametros do MLPClassifier
from src.Mapper import Mapper
from src.env import *

import sys
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")
np.set_printoptions(threshold=sys.maxsize)

# funcao que utiliza o metodo train_test_split do sklearn para dividir o dataset inicial em folds de treino e teste
# estamos gravando esse resultado em csvs para que os valores não mudem a cada execucao do codigo

def generate_csv():
    global train_df, test_df, train_targets, test_targets
    ionosphere_df = pd.read_csv('ionosphere.data', header=None)
    # # dividir o dataset
    ionosphere_x_df = ionosphere_df.drop(labels=34, axis=1)
    ionosphere_y_df = ionosphere_df.iloc[:, -1].to_frame()
    ionosphere_y_df = ionosphere_y_df.replace('g', 1).replace('b', 0)
    # usando holdout 60-40
    # se atentar se a cada execucao do codigo, os mesmos dados sao repartidos
    train_df, test_df, train_targets, test_targets = train_test_split(ionosphere_x_df, ionosphere_y_df, test_size=0.3,
                                                                      stratify=ionosphere_y_df)
    train_df.to_csv('folds\\train_df.csv', index=False, header=False)
    test_df.to_csv('folds\\test_df.csv', index=False, header=False)
    train_targets.to_csv('folds\\train_targets.csv', index=False, header=False)
    test_targets.to_csv('folds\\test_targets.csv', index=False, header=False)


def init_mlp():
    #generate_csv()

    start = datetime.datetime.now()

    # ignoramos a primeira linha dos datasets, pois a conversao
    # para csv considerou o index das colunas como uma nova linha
    
    train_df = pd.read_csv('folds\\train_df.csv', header=None)
    # train_df = train_df.drop(train_df.index[0])

    test_df = pd.read_csv('folds\\test_df.csv', header=None)
    # test_df = test_df.drop(test_df.index[0])

    train_targets = pd.read_csv('folds\\train_targets.csv', header=None)
    # train_targets = train_targets.drop(train_targets.index[0])

    test_targets = pd.read_csv('folds\\test_targets.csv', header=None)
    # test_targets = test_targets.drop(test_targets.index[0])


    # estrategia baseada em grade para testar a rede MLP

    neuronios = [3, 10, 15]
    taxas = [0.6, 0.3, 0.1]
    epocas = [300, 700, 1000]

    for neuronio in neuronios:
        for taxa in taxas:
            for epoca in epocas:
                file_name = f"{neuronio}_{taxa}_{epoca}"
                sys.stdout = open(file_name, "w")
                # taxa de aprendizado adaptativa, talvez tenha que mudar porconta da estrategia em grade

                mlp_model = MLPClassifier(hidden_layer_sizes=neuronio,
                                          max_iter=epoca,
                                          alpha=1e-05,
                                          activation='logistic',
                                          solver='sgd',
                                          learning_rate='adaptive',
                                          learning_rate_init=taxa,
                                          tol=1e-07,
                                          verbose=True)
                mlp_model.fit(train_df, train_targets)

                print('PARAMETROS DE INICIALIZACAO DA REDE\nNUMERO DE NEURONIOS ')
                print(f'Camada de Entrada: 34')
                print(f'Camada Escondida: {mlp_model.hidden_layer_sizes}')
                print(f'Camada de Saida: {mlp_model.n_outputs_}\n')

                print('--- PARAMETROS DE CONFIGURACAO DA REDE ---')
                print(f'Numero de Epocas: {mlp_model.n_iter_}')
                print(f'Taxa de Aprendizado: {mlp_model.learning_rate}')
                print(f'Taxa de Aprendizado Inicial: {mlp_model.learning_rate_init}')

                print('METRICAS\n')
                #predictions_proba = mlp_model.predict_proba(test_df)
                predictions = mlp_model.predict(test_df)
                print("RESULTADOS:\n")
                print(predictions)
                print(f'ACURACIA: {accuracy_score(test_targets, predictions)}\n')
                sys.stdout.close()


init_mlp()

