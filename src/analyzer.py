import re
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
import seaborn as sns
import matplotlib.pyplot as plt

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