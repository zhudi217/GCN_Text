import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import numpy as np
import networkx as nx
from collections import OrderedDict
from itertools import combinations
import math
from tqdm import tqdm


def load_pickle(filename):
    completeName = os.path.join("./data/",
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data


def save_as_pickle(filename, data):
    completeName = os.path.join("./data/",\
                                filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)


def nCr(n,r):
    f = math.factorial
    return int(f(n)/(f(r)*f(n-r)))


# remove stopwords and non-words from tokens list
def filter_tokens(tokens, stopwords):
    tokens1 = []
    for token in tokens:
        if (token not in stopwords) and (token not in [".", ",", ";", "&", "'s", ":", "?", "!", "(", ")", "'", "'m",
                                                       "'no", "***", "--", "...", "[", "]"]):
            tokens1.append(token)
    return tokens1


def dummy_fun(doc):
    return doc


def word_word_edges(p_ij):
    word_word = []
    cols = list(p_ij.columns)
    cols = [str(w) for w in cols]
    for w1, w2 in tqdm(combinations(cols, 2), total=nCr(len(cols), 2)):
        if p_ij.loc[w1, w2] > 0:
            word_word.append((w1, w2, {"weight": p_ij.loc[w1, w2]}))
    return word_word


def generate_text_graph(window=10):
    """ generates graph based on text corpus; window = sliding window size to calculate point-wise mutual information between words """
    print("Preparing data...")
    datafolder = "./data/"
    df = pd.read_csv(os.path.join(datafolder,"t_bbe.csv"))
    df.drop(["id", "v"], axis=1, inplace=True)
    df = df[["t", "c", "b"]]
    stopwords = list(set(nltk.corpus.stopwords.words("english")))
    
    # one chapter per document, labelled by book
    df_data = pd.DataFrame(columns=["c", "b"])
    for book in df["b"].unique():
        dum = pd.DataFrame(columns=["c", "b"])
        dum["c"] = df[df["b"] == book].groupby("c").apply(lambda x: (" ".join(x["t"])).lower())
        dum["b"] = book
        df_data = pd.concat([df_data, dum], ignore_index=True)
    del df
    
    # tokenize & remove funny characters
    df_data["c"] = df_data["c"].apply(lambda x: nltk.word_tokenize(x)).apply(lambda x: filter_tokens(x, stopwords))
    save_as_pickle("df_data.pkl", df_data)
    
    # Tfidf
    print("Calculating Tf-idf...")
    vectorizer = TfidfVectorizer(input="content", max_features=None, tokenizer=dummy_fun, preprocessor=dummy_fun)
    vectorizer.fit(df_data["c"])
    df_tfidf = vectorizer.transform(df_data["c"])
    df_tfidf = df_tfidf.toarray()
    vocab = vectorizer.get_feature_names()
    vocab = np.array(vocab)
    df_tfidf = pd.DataFrame(df_tfidf,columns=vocab)
    
    # PMI between words
    names = vocab
    n_i = OrderedDict((name, 0) for name in names)
    word2index = OrderedDict((name, index) for index, name in enumerate(names))

    occurrences = np.zeros((len(names), len(names)), dtype=np.int32)
    # Find the co-occurrences:
    num_windows = 0
    print("Calculating co-occurences...")
    for l in tqdm(df_data["c"], total=len(df_data["c"])):
        for i in range(len(l)-window):
            num_windows += 1
            d = set(l[i:(i+window)])

            for w in d:
                n_i[w] += 1
            for w1, w2 in combinations(d, 2):
                i1 = word2index[w1]
                i2 = word2index[w2]

                occurrences[i1][i2] += 1
                occurrences[i2][i1] += 1

    print("Calculating PMI...")
    # convert to PMI
    p_ij = pd.DataFrame(occurrences, index=names, columns=names)/num_windows
    p_i = pd.Series(n_i, index=n_i.keys())/num_windows

    del occurrences
    del n_i
    for col in p_ij.columns:
        p_ij[col] = p_ij[col]/p_i[col]
    for row in p_ij.index:
        p_ij.loc[row, :] = p_ij.loc[row, :]/p_i[row]
    p_ij = p_ij + 1E-9
    for col in p_ij.columns:
        p_ij[col] = p_ij[col].apply(lambda x: math.log(x))
        
    # Build graph
    print("Building graph (No. of document, word nodes: %d, %d)..." %(len(df_tfidf.index), len(vocab)))
    G = nx.Graph()
    print("Adding document nodes to graph...")
    G.add_nodes_from(df_tfidf.index)    # document nodes
    print("Adding word nodes to graph...")
    G.add_nodes_from(vocab)     # word nodes
    # build edges between document-word pairs
    print("Building document-word edges...")
    document_word = [(doc, w, {"weight": df_tfidf.loc[doc, w]}) for doc in tqdm(df_tfidf.index, total=len(df_tfidf.index))\
                     for w in df_tfidf.columns]
    
    print("Building word-word edges...")
    word_word = word_word_edges(p_ij)
    save_as_pickle("word_word_edges.pkl", word_word)
    print("Adding document-word and word-word edges...")
    G.add_edges_from(document_word)
    G.add_edges_from(word_word)
    save_as_pickle("text_graph.pkl", G)
    print("Done and saved!")


if __name__ == "__main__":
    generate_text_graph()