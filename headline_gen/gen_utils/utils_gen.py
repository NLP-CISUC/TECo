import string

from sklearn.feature_extraction.text import TfidfVectorizer
import NLPyPort as nlpyport


def get_word_tfidf(prov_id, docs):
    vectorizer = TfidfVectorizer(use_idf=True)
    tfidf_matrix = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names()
    feature_index = tfidf_matrix[prov_id, :].nonzero()[1]
    tfidf_scores = zip(feature_index, [tfidf_matrix[prov_id, x] for x in feature_index])
    return sorted([(feature_names[i], s) for (i, s) in tfidf_scores], key=lambda tup: tup[1], reverse=True)


def get_word_tfidf_v2(headline, all_words, all_occurrences):
    token_dets = []
    hl_tokens = headline.lower().translate(str.maketrans('', '', string.punctuation)).split()
    for token in hl_tokens:
        if token in all_words:
            token_dets.append((token, all_occurrences[all_words.index(token)]))
    return sorted(token_dets, key=lambda tup: tup[1], reverse=False)


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
    labels_append = count_labels.append
    for label in all_labels:
        if label[0] == token:
            labels_append(label)

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


def check_pos(w_pos):
    if w_pos[0] == 'n' and 'letra' not in w_pos:
        return True
    elif 'adj' in w_pos:
        return True
    elif w_pos[0] == 'v':
        return True
    return False


def trim_pos(w_pos):
    while '+' in w_pos:
        w_pos = w_pos.split('+')[0]
    while '-' in w_pos:
        w_pos = w_pos.split('-')[0]
    return w_pos


def get_right_form(keyword_det, sub_det, all_labels):
    if keyword_det[2][0] != sub_det[2][0]:
        return "######"

    if keyword_det[2][0] == 'v':
        tmp = get_right_verb_form(keyword_det, sub_det, all_labels)
        if tmp and tmp != keyword_det[0]:
            # print("\nSUB VERB: ", keyword_det, sub_det, tmp)
            return tmp
        return "######"
    else:
        keyword_form = keyword_det[3].split(":")
        # print(keyword_det, sub_det, keyword_form)
        for form in keyword_form:
            if form.lower().translate(str.maketrans('', '', string.punctuation)).strip() in sub_det[3]:
                return sub_det[0]

        keyword_form = keyword_det[3]
        for label in all_labels:
            # check lemma and form
            if label[1] == sub_det[1] and (label[3] == keyword_form or keyword_form in label[3]):
                return label[0]
    # if a substitute has '######' in it, it's invalid
    # print("######", w, sub, trim_pos(sub[3]))
    return "######"


def get_sentence_vector(tokens, model):
    vectors = []
    append_vectors = vectors.append
    for t in tokens:
        if t in model.vocab:
            append_vectors(model.wv[t])
    if len(vectors) > 0:
        return sum(vectors) / len(vectors)
    else:
        return []


def get_right_verb_form(keyword_det, sub_det, all_labels):
    # print("[VERB]\t", keyword_det, sub_det)
    keyword_form = keyword_det[3].split(":")
    for label in all_labels:
        if label[1] == sub_det[1]:
            sub_form = label[3].split(":")
            for form in sub_form:
                if form in keyword_form and 'y' not in form:
                    # print("[TESTE] ", label, sub_det, keyword_det, form, keyword_form)
                    if 'w' in label[3] and 'z' in label[3]:
                        return label[1]
                    # print("$$$$ \t", label, keyword_det, sub_det)
                    return label[0]
    return []
