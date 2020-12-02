import string

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from proverb_selector.sel_utils.file_manager import *
from sel_approach_we.data_manager import preprocess


def init_prov_selector_standard(algorithm, input_text, proverbs, amount, corpus=None):
    if algorithm == 0:
        vectorizer = TfidfVectorizer(max_df=0.75, min_df=2)
    else:
        vectorizer = CountVectorizer()
    sim = []

    if not corpus:
        corpus = proverbs

    vectorizer.fit_transform(corpus)
    vectors_p = vectorizer.transform(proverbs)
    vec_input = vectorizer.transform([input_text])
    aux_sim = cosine_similarity(vec_input[0:1], vectors_p).tolist()[0]
    sim.extend(aux_sim)

    # for i in input_text:
    #     docs = proverbs.copy()
    #     docs.insert(0, i)
    #     matrix = vectorizer.fit_transform(docs)
    #     aux_sim = cosine_similarity(matrix[0:1], matrix).tolist()[0]
    #     sim.append(aux_sim[1:])

    return selector(input_text, proverbs, sim, amount)


def init_prov_selector_we(input_text, generated, model, corpus=None, tfidf=False, input_tokens=None, amount=10):

    if not model:
        print("[ERROR] No model for WE selection.")
        return None

    if tfidf:
        if not corpus:
            corpus = generated
        vectorizer = TfidfVectorizer(max_df=0.75, min_df=2)
        vectorizer.fit_transform(corpus)

    if not input_tokens:
        input_tokens = input_text.lower().translate(str.maketrans('', '', string.punctuation)).split()

    #print("1) input_tokens: ", input_tokens)
    input_tokens = [t.lower() for t in input_tokens if t.lower() in model.vocab]
    #print("2) input_tokens: ", input_tokens)
    input_vector = get_vector_for_text(input_text, input_tokens, model, tfidf, vectorizer)

    gen_tokens_list = []
    for g in generated:
        g_tokens = g.lower().translate(str.maketrans('', '', string.punctuation)).split()
        g_tokens = [t for t in g_tokens if t in model.vocab]
        gen_tokens_list.append(g_tokens)

    sim = []
    for i, tok_list in enumerate(gen_tokens_list):
        gen_exp_vector = get_vector_for_text(generated[i], tok_list, model, tfidf, vectorizer)
        sim.append(cosine_similarity([gen_exp_vector], [input_vector]).tolist()[0])

    return selector(input_text, generated, sim, amount)

def get_vector_for_text(input_text, input_tokens, model, tfidf, vectorizer):
    '''
    vector = []
    for t in input_tokens:
        print(t in model.vocab)
        print("...", t)
        vector.append(model[t])
    '''

    vectors = [model[t] for t in input_tokens]
    if tfidf:
        weights = get_tfidf_weights(vectorizer, input_text, input_tokens)
        #print("weights", weights)
        for i, tok in enumerate(input_tokens):
            vectors[i] = [v * weights[i] for v in vectors[i]]

    den = len(vectors)
    avg_vec = [sum(x) / den for x in zip(*vectors)]

    return avg_vec


def get_tfidf_weights(vectorizer, doc, doc_tokens):
    doc_vec = vectorizer.transform([doc]).toarray()[0]
    features = vectorizer.get_feature_names()

    weights = []
    for t in doc_tokens:
        if t in features:
            it = features.index(t)
            weights.append(1+doc_vec[it])
        else:
            weights.append(1)

    return weights


#TODO: apagar o que se segue?
def get_sentence_vector(tokens, model, tfidf, inp_id, vectorizer, matrix):
    vectors = [model.vw[t] for t in tokens]
    if tfidf:  # apply tf-idf
        for count_tok, tok in enumerate(tokens):
            tf_idf = get_word_tfidf(tok, inp_id, vectorizer, matrix)
            vectors[count_tok] = vectors[count_tok] * tf_idf
    return sum(vectors) / len(vectors)


def get_word_tfidf(word, inp_id, vectorizer, tfidf_matrix):
    array_names = vectorizer.get_feature_names()
    if word in array_names:
        tmp = array_names.index(word)
        #print(tfidf_matrix.shape)
        # get the first vector out (for the first document)
        first_vector_tfidfvectorizer = tfidf_matrix[inp_id]
        # place tf-idf values in a pandas data frame
        df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=vectorizer.get_feature_names(),
                          columns=["tfidf"])
        df.sort_values(by=["tfidf"], ascending=False)

        return df.values.tolist()[tmp][0]
    else:
        return 0