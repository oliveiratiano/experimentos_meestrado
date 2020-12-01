import string
import unidecode
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import glob
import sys
from tqdm import tqdm
from gensim.models import Word2Vec
from gensim.models import FastText
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import statistics

def encontrar_ranking_modelo_gensim(linha_termo, coluna, modelo):
        lista_termos_coluna = linha_termo[coluna].split("/")
        validos = 0
        lista_rank_presentes = []
        lista_sim_presentes = []
        lista_termos_presentes = []
        lista_termos_ausentes = []
        for termo_relac in lista_termos_coluna:
            #caso o termo que esteja entre as barras seja composto de uma única palavra
            if termo_relac != '' and termo_relac != linha_termo.termo and len(termo_relac.strip().split(' ')) == 1:
                #documenta o total de termos validos na coluna do tesauro
                validos += 1
                try:
                    #adiciona o rank do termo relacionado na lista de ranks
                    lista_rank_presentes.append(modelo.wv.rank(linha_termo.termo.lower(), termo_relac.lower()))
                    lista_sim_presentes.append(modelo.similarity(linha_termo.termo.lower(), termo_relac.lower()))
                    lista_termos_presentes.append(termo_relac)
                except KeyError:
                    lista_termos_ausentes.append(termo_relac)

        return(pd.Series([linha_termo.termo, coluna, lista_rank_presentes, lista_sim_presentes, lista_termos_presentes, lista_termos_ausentes]))

def calcular_modelo_tesauro(tesauro, modelo, coluna):
    tesauro['unico'] = tesauro.apply(lambda x: 1 if x['termo'].strip().count(' ') == 0 and
                                     x[coluna].strip() != ''
                                     else 0, axis = 1)
    termos = tesauro.loc[tesauro.unico == 1].reset_index()
    return termos.apply(lambda x: encontrar_ranking_modelo_gensim(x, coluna, modelo), axis = 1)

def calcular_desempenho(tesauro, modelo, nome_modelo):
    df_use = calcular_modelo_tesauro(tesauro, modelo, 'use')
    df_up = calcular_modelo_tesauro(tesauro, modelo, 'up')
    df_te = calcular_modelo_tesauro(tesauro, modelo, 'te')
    df_tr = calcular_modelo_tesauro(tesauro, modelo, 'tr')
    df_tg = calcular_modelo_tesauro(tesauro, modelo, 'tg')
    df_use.columns = ['termo', 'coluna', 'lista_rank', 'lista_dist', 'lista_presentes', 'lista_ausentes']
    df_up.columns = ['termo', 'coluna', 'lista_rank', 'lista_dist', 'lista_presentes', 'lista_ausentes']
    df_te.columns = ['termo', 'coluna', 'lista_rank', 'lista_dist', 'lista_presentes', 'lista_ausentes']
    df_tr.columns = ['termo', 'coluna', 'lista_rank', 'lista_dist', 'lista_presentes', 'lista_ausentes']
    df_tg.columns = ['termo', 'coluna', 'lista_rank', 'lista_dist', 'lista_presentes', 'lista_ausentes']
    
    display('Rank TSTF ' + nome_modelo + ':')

    rank_medio_use = sum(df_use.lista_rank.values.sum())/len(df_use.lista_rank.values.sum())
    rank_mediano_use = statistics.median(df_use.lista_rank.values.sum())
    qtde_termos_use = len(df_use.lista_rank.values.sum())
    desvio_use = statistics.stdev(df_use.lista_rank.values.sum())
    display('rank médio use: ' + str(rank_medio_use) + 
        ' - rank mediano use: ' + str(rank_mediano_use) + 
        ' - desvio padrão use: ' + str(desvio_use) +
        ' - total: ' + str(qtde_termos_use))

    rank_medio_up = sum(df_up.lista_rank.values.sum())/len(df_up.lista_rank.values.sum())
    rank_mediano_up = statistics.median(df_up.lista_rank.values.sum())
    qtde_termos_up = len(df_up.lista_rank.values.sum())
    desvio_up = statistics.stdev(df_up.lista_rank.values.sum())
    display('rank médio up: ' + str(rank_medio_up) + 
        ' - rank mediano up: ' + str(rank_mediano_up) + 
        ' - desvio padrão up: ' + str(desvio_up) +
        ' - total: ' + str(qtde_termos_up))

    rank_medio_te = sum(df_te.lista_rank.values.sum())/len(df_te.lista_rank.values.sum())
    rank_mediano_te = statistics.median(df_te.lista_rank.values.sum())
    qtde_termos_te = len(df_te.lista_rank.values.sum())
    desvio_te = statistics.stdev(df_te.lista_rank.values.sum())
    display('rank médio te: ' + str(rank_medio_te) + 
        ' - rank mediano te: ' + str(rank_mediano_te) + 
        ' - desvio padrão te: ' + str(desvio_te) +
        ' - total: ' + str(qtde_termos_te))

    rank_medio_tr = sum(df_tr.lista_rank.values.sum())/len(df_tr.lista_rank.values.sum())
    rank_mediano_tr = statistics.median(df_tr.lista_rank.values.sum())
    qtde_termos_tr = len(df_tr.lista_rank.values.sum())
    desvio_tr = statistics.stdev(df_tr.lista_rank.values.sum())
    display('rank médio tr: ' + str(rank_medio_tr) + 
        ' - rank mediano tr: ' + str(rank_mediano_tr) + 
        ' - desvio padrão tr: ' + str(desvio_tr) +
        ' - total: ' + str(qtde_termos_tr))

    rank_medio_tg = sum(df_tg.lista_rank.values.sum())/len(df_tg.lista_rank.values.sum())
    rank_mediano_tg = statistics.median(df_tg.lista_rank.values.sum())
    qtde_termos_tg = len(df_tg.lista_rank.values.sum())
    desvio_tg = statistics.stdev(df_tg.lista_rank.values.sum())
    display('rank médio tg: ' + str(rank_medio_tg) + 
        ' - rank mediano tg: ' + str(rank_mediano_tg) + 
        ' - desvio padrão tg: ' + str(desvio_tg) +
        ' - total: ' + str(qtde_termos_tg))

    mp =  (len(df_use.lista_rank.values.sum()) * rank_medio_use +
           len(df_up.lista_rank.values.sum()) * rank_medio_up +
           len(df_te.lista_rank.values.sum()) * rank_medio_te +
           len(df_tr.lista_rank.values.sum()) * rank_medio_tr +
           len(df_tg.lista_rank.values.sum()) * rank_medio_tg) / (len(df_use.lista_rank.values.sum()) +
                                                                  len(df_up.lista_rank.values.sum()) +
                                                                  len(df_te.lista_rank.values.sum()) +
                                                                  len(df_tr.lista_rank.values.sum()) +
                                                                  len(df_tg.lista_rank.values.sum()))
    display('rank médio podenrado ' + nome_modelo + ': ' + str(mp))
    return([df_use, df_up, df_te, df_tr, df_tg])

def restrict_w2v(w2v, restricted_word_set):
    new_vectors = []
    new_vocab = {}
    new_index2entity = []
    new_vectors_norm = []

    for i in range(len(w2v.vocab)):
        word = w2v.index2entity[i]
        vec = w2v.vectors[i]
        vocab = w2v.vocab[word]
        vec_norm = w2v.vectors_norm[i]
        if word in restricted_word_set:
            vocab.index = len(new_index2entity)
            new_index2entity.append(word)
            new_vocab[word] = vocab
            new_vectors.append(vec)
            new_vectors_norm.append(vec_norm)

    w2v.vocab = new_vocab
    w2v.vectors = np.array(new_vectors)
    w2v.index2entity = new_index2entity
    w2v.index2word = new_index2entity
    w2v.vectors_norm = new_vectors_norm


def plotar_dists(lista_manipulos):
    lista_rotulos = ['Distribuição de distâncias USE', 
                'Distribuição de distâncias UP',
                'Distribuição de distâncias TE',
                'Distribuição de distâncias TR',
                'Distribuição de distâncias TG']
    lista_completa = []

    fig, axs = plt.subplots(2, 3, figsize=(20, 12))

    for i, manipulo in enumerate(lista_manipulos):
        if i <= 2:
            lin = 0
            col = i
        else:
            lin = 1
            col = i-3    

        lista_completa += manipulo.lista_dist.sum()
        sns.distplot(manipulo.lista_dist.sum(), ax=axs[lin, col]).set(xlim=(-1,1), ylim = (0,2.5))
        mean = np.mean(manipulo.lista_dist.sum())
        axs[lin, col].axvline(mean, color='r', linestyle='--')
        axs[lin, col].text(mean+0.03,0.2,str(round(mean, 2)),rotation=0, color = 'r')
        axs[lin, col].set_title(lista_rotulos[i])

    sns.distplot(lista_completa, ax=axs[1, 2]).set(xlim=(-1,1), ylim = (0,2.5))
    mean = np.mean(lista_completa)
    axs[1, 2].axvline(mean, color='r', linestyle='--')
    axs[1, 2].text(mean+0.03,0.2,str(round(mean, 2)),rotation=0, color = 'r')
    axs[1, 2].set_title('Distribuição completa de distâncias')

    for ax in axs.flat:
        ax.set(xlabel='Similaridade de cosseno', ylabel='Proporção %')