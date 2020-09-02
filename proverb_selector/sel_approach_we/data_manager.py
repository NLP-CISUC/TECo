import NLPyPort as nlpyport
from proverb_selector.sel_utils.file_manager import get_word_tfidf


def get_sentence_vector(tokens, model, aux_tfidf, inp_id, vectorizer, matrix):
    vectors = [model.wv[t] for t in tokens]
    if aux_tfidf:       # apply tf-idf
        for count_tok, tok in enumerate(tokens):
            tf_idf = get_word_tfidf(tok, inp_id, vectorizer, matrix)
            vectors[count_tok] = vectors[count_tok] * tf_idf
    return sum(vectors)/len(vectors)


def preprocess(txt, vocab, is_str):
    unwanted_chars = ['\n', '.', ',', ';']
    if is_str:
        tokens = nlpyport.tokenize_from_string(txt)
        tags = nlpyport.tag(tokens)[1][0]
    else:
        tokens = txt
        tags = nlpyport.tag(txt)[1][0]

    tmp_tok = []
    for count_tok, tok in enumerate(tokens):
        if tok not in unwanted_chars and tok in vocab:
            if tags[count_tok][1] != 'N':
                tmp_tok.append(tok.lower())
            else:
                tmp_tok.append(tok)
    tmp_tok = [a for a in tmp_tok if a != []]
    return tmp_tok


def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


def cos_sim(inp_vec, txt_vec, model):
    return max(model.wv.cosine_similarities(inp_vec, [txt_vec]))


