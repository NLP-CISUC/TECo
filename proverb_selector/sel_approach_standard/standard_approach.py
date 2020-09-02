from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from proverb_selector.sel_utils.file_manager import *


def init_prov_selector_standard(algorithm, input_text, proverbs, amount):
    info_sim = ''
    if algorithm == 0:
        info_sim = 'STANDARD_TFIDF'
        vectorizer = TfidfVectorizer()
    elif algorithm == 1:
        info_sim = 'STANDARD_COUNTVECTORIZER'
        vectorizer = CountVectorizer()
    sim = []
    for i in input_text:
        docs = proverbs.copy()
        docs.insert(0, i)
        matrix = vectorizer.fit_transform(docs)
        aux_sim = cosine_similarity(matrix[0:1], matrix).tolist()[0]
        sim.append(aux_sim[1:])

    return selector(input_text, proverbs, sim, info_sim, amount)
