from proverb_selector.sel_approach_we.data_manager import *
from proverb_selector.sel_approach_we.we_object_manager import *
from proverb_selector.sel_utils.file_manager import *


def init_prov_selector_we(input_text, proverbs, algorithm, model_fn, amount):
    # OBJECT CREATION
    # proverbs, news_tokens, news_tags, prov_tokens, prov_tags, prov_lemmas  = objects_creation()
    # model = KeyedVectors.load_word2vec_format(
    # '../../models_db/we_models/fasttext_cbow_s300.txtfasttext_cbow_s300.txt')

    # READ TOKENS & PROVERBS
    news_tokens, news_tags, prov_tokens, prov_tags = read_objects()

    # WRITE TOKENS & PROVERBS
    # write_objects(proverbs, news_tokens, news_tags, prov_tokens, prov_tags)
    # model.save('../../models_db/we_models/model_fasttext300_cbow.model')

    sim = [[] for i in input_text]
    info_sim = ""

    # SIMILARITIES
    if algorithm == 2:  # Jaccard - W2V
        model = KeyedVectors.load(model_fn)
        inp_keywords = [preprocess(i.lower(), model.wv.vocab, True) for i in input_text]

        for token in prov_tokens:
            for c_i, i in enumerate(inp_keywords):
                sim[c_i].append(jaccard_similarity(i, token))
        info_sim = 'JACCARD'

    elif algorithm in [3, 4, 5, 6]:  # Cosine - W2V
        vectorizer = None
        matrix = None
        if algorithm in [4, 6]:
            info_sim = 'COSINE_WE_TFIDF_' + model_fn.split("/")[-1].split(".")[0]
            aux_tfidf = True
            vectorizer, matrix = get_tfidf_matrix(input_text, proverbs)
        else:
            info_sim = 'COSINE_WE_' + model_fn.split("/")[-1].split(".")[0]
            aux_tfidf = False

        model = KeyedVectors.load(model_fn)
        inp_keywords = [preprocess(i.lower(), model.wv.vocab, True) for i in input_text]
        prov_keywords = [preprocess(p, model.wv.vocab, False) for p in prov_tokens]

        inp_vec = [get_sentence_vector(i, model, aux_tfidf, inp_id, vectorizer, matrix) for inp_id, i in enumerate(inp_keywords)]

        for count_prov, proverb in enumerate(prov_keywords):
            prov_vec = get_sentence_vector(prov_keywords[count_prov], model, False, count_prov, vectorizer, matrix)
            for c_i, i in enumerate(inp_vec):
                sim[c_i].append(cos_sim(i, prov_vec, model))
    return selector(input_text, proverbs, sim, info_sim, amount)
