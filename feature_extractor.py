import pandas as pd

def same_words_fraction(entry):
    w1 = set(map(lambda word: word.lower().strip(), entry['question1'].split(' ')))
    w2 = set(map(lambda word: word.lower().strip(), entry['question2'].split(' ')))

    return len(w1 & w2)/(len(w1) + len(w2))
