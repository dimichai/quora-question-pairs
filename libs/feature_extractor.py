import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def same_words_fraction(entry):
    w1 = set()
    w2 = set()
    
    if (entry['question1']):
        w1 = set(map(lambda word: word.lower().strip(), entry['question1'].split(' ')))
    if (entry['question2']):
        w2 = set(map(lambda word: word.lower().strip(), entry['question2'].split(' ')))

    return len(w1 & w2)/(len(w1) + len(w2))


def get_cosine_sim(*strs): 
    vectors = [t for t in get_vectors(*strs)]
    return cosine_similarity(vectors)
    
def get_vectors(*strs):
    text = [t for t in strs]
    vectorizer = CountVectorizer(text)
    vectorizer.fit(text)
    return vectorizer.transform(text).toarray()

def get_jaccard_sim(entry):
    q1 = entry['question1']
    q2 = entry['question2']

    a = set(q1.split()) 
    b = set(q2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
