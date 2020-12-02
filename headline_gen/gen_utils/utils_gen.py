import string
import random

from sklearn.feature_extraction.text import TfidfVectorizer
import NLPyPort as nlpyport


def get_tokens(input_text):
    input_text = input_text.replace("«", "").replace("»", "").lower()
    return input_text.translate(str.maketrans('', '', string.punctuation)).split()


def get_word_tfidf(prov_id, docs):
    vectorizer = TfidfVectorizer(use_idf=True)
    tfidf_matrix = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names()
    feature_index = tfidf_matrix[prov_id, :].nonzero()[1]
    tfidf_scores = zip(feature_index, [tfidf_matrix[prov_id, x] for x in feature_index])
    return sorted([(feature_names[i], s) for (i, s) in tfidf_scores], key=lambda tup: tup[1], reverse=True)


def get_word_tfidf_v2(input_text, tfidf, input_tokens=None):
    token_dets = []
    if not input_tokens:
        input_tokens = input_text.lower().translate(str.maketrans('', '', string.punctuation)).split()
    for token in input_tokens:
        if token in tfidf:
            token_dets.append((token, tfidf[token]))
    return sorted(token_dets, key=lambda tup: tup[1], reverse=True)


def find_prov_index(prov, proverbs):
    for c, p in enumerate(proverbs):
        if prov in p.lower().translate(str.maketrans('', '', string.punctuation)):
            return c
    return -1


def find_index(keyword, tuple_list):
    if not tuple_list:
        return -1
    for k_id, tmp_det in enumerate(tuple_list):
        # print(tmp_det)
        if keyword == tmp_det[0]:
            return k_id
    return -1


def find_label(token, list_tokens, all_labels):
    count_labels = []
    if token in all_labels:
        count_labels.extend(all_labels[token])

    if len(count_labels) == 1:      # only one possibility
        return count_labels[0]
    elif not count_labels:          # did not find any correspondent word
        return ()
    else:                           # in case the PoS tag is ambiguous
        tags = nlpyport.tag(list_tokens)
        # print(tags)
        for t in tags[1][0]:
            if t[0] == token:
                for label in count_labels:
                    if label[2][0] == t[1][0] and check_pos(trim_pos(label[2])):
                        return label
        return ()


'''
Checks if open-class PoS
'''
def check_pos(w_pos):
    if w_pos[0] == 'n' and 'letra' not in w_pos:
        return True
    elif 'adj' in w_pos:
        return True
    elif w_pos[0] == 'v':
        return True
    return False

def aux_verb(label):
    return label[2][0] == 'v' and (label[1] == 'ser' or label[1] == 'estar' or label[1] == 'ter' or label[1] == 'ir' or label[1] == 'fazer')

def trim_pos(w_pos):
    while '+' in w_pos:
        w_pos = w_pos.split('+')[0]
    while '-' in w_pos:
        w_pos = w_pos.split('-')[0]
    return w_pos


def get_right_form(keyword_det, sub_det, dict_lemmas_labels):
    if keyword_det[2][0] != sub_det[2][0]:
        return None

    if keyword_det[2][0] == 'v':
        tmp = get_right_verb_form(keyword_det, sub_det, dict_lemmas_labels)
        #Se forma for igual...
        #if tmp and tmp != keyword_det[0]:
            # print("\nSUB VERB: ", keyword_det, sub_det, tmp)
        return tmp
    else:
        keyword_form = keyword_det[3].split(":")
        # print(keyword_det, sub_det, keyword_form)

        # so that it is not always the same
        random.shuffle(keyword_form)
        for form in keyword_form:
            if form.lower().translate(str.maketrans('', '', string.punctuation)).strip() in sub_det[3]:
                return sub_det[0]

        keyword_form = keyword_det[3]
        if sub_det[1] in dict_lemmas_labels:
            for label in dict_lemmas_labels[sub_det[1]]:
                # check lemma and form
                if label[3] == keyword_form or keyword_form in label[3]:
                    return label[0]

    return None


def get_sentence_vector(tokens, model):
    vectors = []
    for t in tokens:
        if t in model.vocab:
            vectors.append(model.wv[t])
    if len(vectors) > 0:
        return sum(vectors) / len(vectors)
    else:
        return []

def get_right_verb_form(keyword_det, sub_det, dict_lemmas_labels):
    # print("[VERB]\t", keyword_det, sub_det)
    keyword_form = keyword_det[3].split(":")
    if sub_det[1] in dict_lemmas_labels:
        for label in dict_lemmas_labels[sub_det[1]]:
            sub_form = label[3].split(":")
            for form in sub_form:
                if form in keyword_form and 'y' not in form:
                    # print("[TESTE] ", label, sub_det, keyword_det, form, keyword_form)
                    if 'w' in label[3] and 'z' in label[3]:
                        return label[1]
                    # print("$$$$ \t", label, keyword_det, sub_det)
                    return label[0]
    return []
