from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from proverb_selector.sel_utils.file_manager import *


def init_prov_selector_standard(algorithm, input_text, proverbs, amount, corpus=None):
    info_sim = None
    if algorithm == 0:
        info_sim = 'STANDARD_TFIDF'
        vectorizer = TfidfVectorizer(max_df=0.75, min_df=2)
    else:
        info_sim = 'STANDARD_COUNTVECTORIZER'
        vectorizer = CountVectorizer()
    sim = []

    if not corpus:
        corpus = proverbs

    vectorizer.fit_transform(corpus)
    for i in input_text:
        vectors_p = vectorizer.transform(proverbs)
        vector_input = vectorizer.transform([i])
        aux_sim = cosine_similarity(vector_input[0:1], vectors_p).tolist()[0]
        sim.append(aux_sim)

    # for i in input_text:
    #     docs = proverbs.copy()
    #     docs.insert(0, i)
    #     matrix = vectorizer.fit_transform(docs)
    #     aux_sim = cosine_similarity(matrix[0:1], matrix).tolist()[0]
    #     sim.append(aux_sim[1:])

    #print(sim)

    return selector(input_text, proverbs, sim, info_sim, amount)
