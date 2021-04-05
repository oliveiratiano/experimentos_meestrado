import re
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, precision_recall_fscore_support, homogeneity_completeness_v_measure

def converter_string_array(str_arr):
    str_arr = re.sub(' +', 
           ' ', 
           str_arr.replace('\n', '').replace('[', '').replace(']', '').replace('  ', ' '))
    arr = np.fromstring(str_arr, sep=' ')
    return(arr.reshape(1, -1).reshape(1,-1))

def calc_matriz_sim(vetores, dir_experimento):
    print("calculando matriz de similaridade nos vetores "+vetores.name)
    sim_m = np.empty(shape=[vetores.shape[0], vetores.shape[0]])
    for i in tqdm(range(0, vetores.shape[0])):
        for j in range(0, vetores.shape[0]):
            a, b = vetores[i], vetores[j].T
            sim_m[i][j] = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    np.save('dados/'+dir_experimento+'/sim_docs_'+vetores.name+'.npy', sim_m)
    return(sim_m)

def calcular_sim_assuntos(assuntos, sim_docs, modelo, dir_experimento):
    print('calculando a similaridade entre assuntos para o modelo '+modelo)
    lista_sim_assuntos = []
    lista_assuntos = assuntos.unique()

    for i in tqdm(range(0, lista_assuntos.shape[0])):
        for j in range(0, lista_assuntos.shape[0]):    
            assunto_a = lista_assuntos[i]
            assunto_b = lista_assuntos[j]

            indices_a = assuntos[assuntos == assunto_a].index.values
            indices_b = assuntos[assuntos == assunto_b].index.values
            x = sim_docs[np.ix_(indices_a,indices_b)]

            #Se os assuntos forem os mesmos, apenas o triângulo superior (sem a diagonal principal) deve ser consideradono cálculo da média
            #caso contrário, todos os elementos podem ser considerados
            if assunto_a == assunto_b:
                ind_sup = np.triu_indices(max(len(indices_a), len(indices_b)), k=1)
                sim = x[ind_sup].mean()
            else:
                sim = sim_docs[np.ix_(indices_a,indices_b)].mean()
            lista_sim_assuntos.append((assunto_a, assunto_b, sim))
            
    plt.rcParams["figure.figsize"] = (50,50)
    lista_sim_assuntos = pd.DataFrame.from_records(lista_sim_assuntos, columns = ['assunto_a', 'assunto_b', 'sim_cos'])
    pivot = lista_sim_assuntos.pivot(index='assunto_a', columns='assunto_b', values='sim_cos')
    fig = sns.heatmap(pivot).get_figure()
    fig.savefig('dados/'+dir_experimento+'/sim_assuntos_'+modelo+'.png', dpi=300)
    plt.cla()
    return(pivot)

def computar_scores_agrupamento(X, y, dir_experimento, modelo, lista_k):
    lista_scores_k = []
    for k in tqdm(lista_k):
        if k == 1:
            sil_score = 0
            ari = 0
            hcv = [0,0,0]
        else:
            kmeans = KMeans(n_clusters=k, random_state=0, verbose = 0).fit(X)
            preds_kmeans = kmeans.predict(X)
            sil_score_kmeans = silhouette_score(X, preds_kmeans)
            ari_kmeans = adjusted_rand_score(y, preds_kmeans)
            hcv_kmeans = homogeneity_completeness_v_measure(y, preds_kmeans)
        lista_scores_k.append((k, 
                              sil_score_kmeans, 
                              ari_kmeans, 
                              hcv_kmeans[0],
                              hcv_kmeans[1], 
                              hcv_kmeans[2]))
    return(lista_scores_k)

def gerar_graficos_kmeans(lista_scores_k, dir_experimento, modelo):
    k_vals = [lista_scores_k[i][0] for i in range(1, len(lista_scores_k))]
    sil_scores = [lista_scores_k[i][1] for i in range(1, len(lista_scores_k))]
    ari_vals = [lista_scores_k[i][2] for i in range(1, len(lista_scores_k))]
    homog_vals = [lista_scores_k[i][3] for i in range(1, len(lista_scores_k))]
    complet_vals = [lista_scores_k[i][4] for i in range(1, len(lista_scores_k))]
    vscore_vals = [lista_scores_k[i][5] for i in range(1, len(lista_scores_k))]

    ax = sns.lineplot(k_vals, 
                    sil_scores, 
                    color = 'g', 
                    label = 'Silhueta', 
                    legend = False)    
    sns.lineplot(k_vals, 
                ari_vals,  
                label = 'ARI', 
                color = 'b', 
                legend = False)
    ax = sns.lineplot(k_vals, 
                    vscore_vals, 
                    color = 'r', 
                    label = 'V_score', 
                    legend = False)
    ax = sns.lineplot(k_vals, 
                    homog_vals, 
                    color = 'k', 
                    label = 'Homogeneidade', 
                    legend = False)
    ax = sns.lineplot(k_vals, 
                    complet_vals, 
                    color = 'y', 
                    label = 'Completude', 
                    legend = False)
    ax.set(xlabel='k', ylabel='Valor')
    ax.grid(False)

    x_vline_ari = k_vals[ari_vals.index(max(ari_vals))]
    x_vline_sil = k_vals[sil_scores.index(max(sil_scores))]
    x_vline_v =  k_vals[vscore_vals.index(max(vscore_vals))]
    x_vline_h =  k_vals[homog_vals.index(max(homog_vals))]
    x_vline_c =  k_vals[complet_vals.index(max(complet_vals))]

    ax.axvline(x_vline_ari, color = 'royalblue', linestyle = '--')
    ax.axvline(x_vline_sil, color = 'seagreen', linestyle = '--')
    ax.axvline(x_vline_v, color = 'indianred', linestyle = '--')
    ax.axvline(x_vline_h, color = 'dimgray', linestyle = '--')
    ax.axvline(x_vline_c, color = 'olive', linestyle = '--')

    ax.text(x_vline_ari+1, .01, x_vline_ari, color = 'royalblue')
    ax.text(x_vline_ari+3, max(ari_vals), round(max(ari_vals), 2), color = 'royalblue')
    ax.text(x_vline_sil+1, 0.01, x_vline_sil, color = 'seagreen')
    ax.text(x_vline_sil+3, max(sil_scores), round(max(sil_scores), 2), color = 'seagreen')
    ax.text(x_vline_v+1, 0.01, x_vline_v, color = 'indianred')
    ax.text(x_vline_v+3, max(vscore_vals), round(max(vscore_vals), 2), color = 'indianred')
    ax.text(x_vline_h+1, 0.01, x_vline_h, color = 'dimgray')
    ax.text(x_vline_h+3, max(homog_vals), round(max(homog_vals), 2), color = 'dimgray')
    ax.text(x_vline_c+1, 0.01, x_vline_c, color = 'olive')
    ax.text(x_vline_c+3, max(complet_vals), round(max(complet_vals), 2), color = 'olive')
    ax.set(ylim=(0, .7))

    lgd = ax.figure.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize='small')

    ax.get_figure().savefig('dados/'+dir_experimento+'/kmeans_'+modelo+'.png', dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()