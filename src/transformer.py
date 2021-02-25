from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import normalize
from gensim.models import Word2Vec
from gensim.models import FastText
from gensim.models.word2vec import LineSentence
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import multiprocessing
import nltk
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
import logging
import subprocess
import os
import time

#recebe o id de um documento e o diretorio onde ele se encontra, como strings
#retorna o texto contido neste documento
def recuperar_teor(idx, diretorio):
    with open(diretorio + idx, "r", encoding='utf-8') as f:
        contents = f.read()
    return contents

#recebe como entrada o dataframe do conjunto de treinamento contendo id, teores e assunto
#retorna conjunto com termos cujo ICA é maior que a média observada no vocabulário extraído dos teores
def sel_termos_ica(X_treino):
    #agrupa teores por assunto, concatenando-os
    df_assuntos = pd.DataFrame()
    for classe in tqdm(X_treino['assunto'].unique()):
        concat = ' '.join(X_treino.loc[X_treino['assunto'] == classe].teores)
        df_assuntos = pd.concat([df_assuntos, pd.DataFrame([(classe, concat)])], ignore_index = True)
    df_assuntos.columns = ['assuntos', 'teores']
    print("-processando strings do corpus")
    #calcula o ICA dos termos que aparecem no conjunto de treino
    stopwords = nltk.corpus.stopwords.words('portuguese')
    df_assuntos['teores'] = df_assuntos.teores.str.replace('\n', ' ').str.strip()
    print("-treinando vetorizador")
    vectorizer = TfidfVectorizer(stop_words = stopwords, max_df = 50, smooth_idf = False)
    vec = vectorizer.fit(df_assuntos.teores)
    
    #retorna conjunto com termos cujo ICA é maior que a média de ICA do vocabulário
    df = pd.DataFrame(list(zip([k for k, v in vec.vocabulary_.items()], vec.idf_)), columns = ['termo', 'idf']).sort_values(by='idf')
    estats_idf = pd.DataFrame(df.idf.describe())
    corte_idf = estats_idf.loc['mean',:]
    df = df[df.idf >= corte_idf[0]]
    print("-ICA processado")
    return(set(df.termo))

#recebe uma série do pandas contendo o corpus, o valor mínimo de frequência das palavras e um conjunto de stopwords
#retorna um conjunto de palavras que aparecem no mínimo freq_min vezes no corpus
def sel_termos_freq(corpus, freq_min, stopwords):
    contagem = corpus.teores.str.split(expand=True).stack().value_counts()
    p_validas = set(contagem.index) - set(stopwords)
    contagem = contagem.loc[contagem.index.intersection(p_validas)][contagem>=freq_min]
    return(set(contagem.index))

#retorna os termos do tstf compostos por apenas uma palavra
def sel_termos_tesauro():
    tesauro = pd.read_csv("tesauro_stf.csv")
    tesauro['unico'] = tesauro.apply(lambda x: 1 if x['termo'].strip().count(' ') == 0
                                        else 0, axis = 1)
    termos_tesauro = tesauro.loc[tesauro.unico == 1].reset_index().termo.str.lower()
    return(set(termos_tesauro))

#recebe o corpus, a frequência mínima de palavras, e um conjunto de stopwords
#retorna um conjunto com o vocabulário
def extrair_vocabulario(corpus, corte_freq, stopwords):
    print("extraindo termos com base no ICA")
    termos_ica = sel_termos_ica(corpus)
    print("extraindo termos com base na frequência - geralmente leva menos de 4 minutos")
    termos_freq = sel_termos_freq(corpus, corte_freq, stopwords)
    print("extraindo termos do tesauro")
    termos_tesauro = sel_termos_tesauro()
    print("extração de vocabulário concluída!")
    vocabulario = termos_tesauro.union(termos_ica).union(termos_freq)
    return vocabulario

def treinar_word2vec(corpus, exp):    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    print("treinando modelo word2vec")
    model = Word2Vec(LineSentence(corpus), size=100, window=5,                 
                 min_count=5, sg=1, hs=1, iter=10, workers=multiprocessing.cpu_count(), sample = 0.00001)
    model.save("dados/experimento_" + str(exp) + "/w2v_jur.model")
    return model

def treinar_fasttext(corpus, exp):    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    print("treinando modelo word2vec")
    model = FastText(size=100, window=5,                 
                 min_count=5, sg=1, hs=1, iter=10, workers=multiprocessing.cpu_count())
    model.build_vocab(corpus_file=corpus)
    total_words = model.corpus_total_words
    model.train(corpus_file=corpus, total_words=total_words, epochs=5)
    model.save("dados/experimento_" + str(exp) + "/ftt_jur.model")
    return model

    
def treinar_glove(exp):    
    print("treinando modelo glove")
    corpus="../mestrado/experimentos_mestrado/dados/experimento_"+str(exp)+"/base_treino_glv.txt"
    vocab_file="../mestrado/experimentos_mestrado/dados/experimento_"+str(exp)+"/glove_vocab.txt"
    coocurrence_file="../mestrado/experimentos_mestrado/dados/experimento_"+str(exp)+"/glv_concurrence.bin"
    coocurrence_shuf_file="../mestrado/experimentos_mestrado/dados/experimento_"+str(exp)+"/glv_concurrence_shuf.bin"
    save_file="../mestrado/experimentos_mestrado/dados/experimento_"+str(exp)+"/glv_jur"
    vector_size=100
    treinar_glove = subprocess.Popen(["bash", "src/glove.sh", corpus, vocab_file, coocurrence_file, coocurrence_shuf_file, save_file, str(vector_size)], 
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    output, errors = treinar_glove.communicate()
    treinar_glove.wait()

    print(output)
    print(errors)
    print("treinamento concluído")

    glove_file = 'dados/experimento_'+str(exp)+'/glv_jur.txt'
    tmp_file = get_tmpfile("test_word2vec.txt")
    _ = glove2word2vec(glove_file, tmp_file)
    model = KeyedVectors.load_word2vec_format(tmp_file)
    return(model)

def remover_stopwords(texto, stopwords):
    tokens = texto.split(" ")
    tokens_filtrados = [p for p in tokens if not p in stopwords]
    return (" ").join(tokens_filtrados).strip()

#recebe uma serie com documentos, um modelo gensim e um conjunto com o vocabulário
#retorna um vetor numpy com a soma dos vetores dos termos dos documentos que estão vocabulário
def calc_vet_soma(serie_documentos, modelo, vocab):
    lista_vecs = []
    for documento in tqdm(serie_documentos):
        lista_tokens = documento.split(' ')
        vetor_doc = np.zeros(100).reshape(1, -1)                                                    
        for token in lista_tokens:
            if token in vocab:
                try:
                    vetor_doc += modelo.wv.get_vector(token)
                except KeyError:
                    pass
        lista_vecs.append(normalize(vetor_doc.reshape(1, -1)))
    return(lista_vecs)


def criar_representacoes_soma(X_teste, y_teste, vocab, diretorio, w2v_jur, ftt_jur, glv_jur, w2v_geral, ftt_geral, glv_geral, exp):
    base_teste = pd.DataFrame(X_teste)
    base_teste['id'] = base_teste.id + '.txt'
    print("recuperando teores da base de teste")
    base_teste['teores'] = [recuperar_teor(x, diretorio) for x in tqdm(base_teste.id)]
    base_teste['assunto'] = y_teste
    docs_teste = base_teste.teores.reset_index().teores

    print("criando representações word2vec geral")
    base_teste['vec_w2v_ger_soma'] = calc_vet_soma(docs_teste, w2v_geral, vocab)
    print("criando representações fasttext geral")
    base_teste['vec_ftt_ger_soma'] = calc_vet_soma(docs_teste, ftt_geral, vocab)
    print("criando representações glove geral")
    base_teste['vec_glv_ger_soma'] = calc_vet_soma(docs_teste, glv_geral, vocab)
    print("criando representações word2vec juridico")
    base_teste['vec_w2v_jur_soma'] = calc_vet_soma(docs_teste, w2v_jur, vocab)
    print("criando representações fasttext juridico")
    base_teste['vec_ftt_jur_soma'] = calc_vet_soma(docs_teste, ftt_jur, vocab)
    print("criando representações glove juridico")
    base_teste['vec_glv_jur_soma'] = calc_vet_soma(docs_teste, glv_jur, vocab)
    base_teste.to_csv('dados/experimento_'+str(exp)+'/vetores_teste.csv', index=False)
    print("----------- EXPERIMENTO "+ str(exp) + " CONCLUIDO -----------")

def criar_repr_agreg(X_treino, X_teste, y_treino, y_teste, vocab, diretorio, exp):
    # treinando modelos de dominio juridico
    w2v_jur = treinar_word2vec('dados/experimento_'+str(exp)+'/base_treino.txt', exp)
    ftt_jur = treinar_fasttext('dados/experimento_'+str(exp)+'/base_treino.txt', exp)
    glv_jur = treinar_glove(exp)
    
    # importando modelos de domínio geral
    w2v_geral = KeyedVectors.load_word2vec_format('modelos/w2v_skip_nilc.txt')
    ftt_geral = KeyedVectors.load_word2vec_format('modelos/ftt_skip_nilc.txt')
    glv_geral = KeyedVectors.load_word2vec_format('modelos/glove_nilc.txt')
    
    #criando representações através da soma de vetores
    criar_representacoes_soma(X_teste, y_teste, vocab, diretorio, w2v_jur, ftt_jur, glv_jur, w2v_geral, ftt_geral, glv_geral, exp)

#recebe o número do experimento e os ids da base de treino
#cria os arquivos necessários para o treinamento dos modelos
#retorna a base de treino enriquecida dos teores e dos assuntos
def criar_base_treino(exp, X_treino, y_treino, diretorio, stopwords):
    X_treino = pd.DataFrame(X_treino)
    X_treino['id'] = X_treino.id + '.txt'


    print("criando base de treino para o experimento "+str(exp))
    if not os.path.exists('dados/experimento_'+str(exp)):
        os.makedirs('dados/experimento_'+str(exp))

    #a base de treino para o word2vec e fasttext deve ter uma frase por linha
    #a base de treino para o glove deve ter um documento por linha
    base_treino = open('dados/experimento_'+str(exp)+'/base_treino.txt', 'w+', encoding='utf8')
    base_treino_glv = open('dados/experimento_'+str(exp)+'/base_treino_glv.txt', 'w+', encoding='utf8')
    tokens = 0
    for documento in tqdm(X_treino.id.values):
        doc = open(diretorio + documento, 'r', encoding='utf8')
        for frase in doc:
            base_treino.write(frase)
            tokens += len(frase.split(" "))
        doc.close()

        doc = open(diretorio + documento, 'r', encoding='utf8')
        teor_completo = doc.read().replace('\n', '')
        teor_completo = remover_stopwords(teor_completo, stopwords)
        base_treino_glv.write(teor_completo + '\n')
        doc.close()

    base_treino.close()
    base_treino_glv.close()
    print(str(tokens)+ " tokens copiados com sucesso")

    print("preparando documentos para extração do vocabulário:")
    X_treino['teores'] = [recuperar_teor(x, diretorio) for x in tqdm(X_treino.id)]
    X_treino['assunto'] = y_treino
    return X_treino

#recebe série com ids dos documentos válidos e a quantidade de experimentos que deve ser executada
#retorna os documentos da base de testes representados como vetores por cada um dos modelos no diretório de dados
def transform(documentos_validos, n_experimentos):
    sss = StratifiedShuffleSplit(n_splits=n_experimentos, test_size=0.2, random_state=0)
    X = documentos_validos.id
    y = documentos_validos.Assunto
    stopwords = nltk.corpus.stopwords.words('portuguese')
    diretorio = "dados/corpus_tratado/"

    #index[0] são os indices de treino, e index[1] são os de teste
    #i é o código do experimento
    for i, index in enumerate(sss.split(X, y)):    
        exp = i+1
        print("----------------------- EXPERIMENTO "+ str(exp) + " -----------------------")
        start = time.time()
        X_treino, X_teste = X[index[0]], X[index[1]]
        y_treino, y_teste = y[index[0]], y[index[1]]  

        # instanciando o corpus do conjunto de treinamento
        base_treino = criar_base_treino(exp, X_treino, y_treino, diretorio, stopwords)

        # criando vocabulário
        freq_min = 100
        vocab = extrair_vocabulario(base_treino, freq_min, stopwords)

        # criando representações para cada experimento
        vetores_teste = criar_repr_agreg(X_treino, X_teste, y_treino, y_teste, vocab, diretorio, exp)        
        end = time.time()
        print('tempo do experimento: ' + str((end - start)/60) +' minutos')