from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from gensim.models import FastText
from gensim.models.word2vec import LineSentence
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import multiprocessing
import nltk
import pandas as pd
from tqdm import tqdm
import logging
import subprocess

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