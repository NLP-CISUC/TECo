"""
@author: Rui Mendes
"""
import random
import re

from proverb_selector.sel_approach_standard.standard_approach import init_prov_selector_standard
from proverb_selector.sel_approach_transformer.transformer_approach import init_prov_selector_bert
from proverb_selector.sel_approach_we.we_approach import init_prov_selector_we
from gen_methods.selection_methods import get_first_selection
from gen_methods.substitution_methods import *
from teco_twitterbot.twitter_utils.twitter_manager import get_info_methods
from gen_utils.utils_gen import *


def get_best_keywords(sentence_words, all_labels, model, order):
    """
    This function chooses the most important words from a sentence from their Tf-Idf value.
    @:param sentence_words: list of tuples, each containing a word from the sentence and its Tf-Idf value
    @:param order: if True: descending order, if False: ascending order
    @:returns list of tuples with the structures representing the two most valuable keywords.
    """
    list_keywords = []
    append_keywords = list_keywords.append
    sentence_tokens = [w[0] for w in sentence_words]
    for word, similarity in sentence_words:
        word_det = find_label(word, sentence_tokens, all_labels)
        if word_det == () or not check_pos(word_det[2]) or word_det[0] not in model.vocab:
            continue
        append_keywords((word_det[0], word_det[1], word_det[2], word_det[3], similarity))

    if not list_keywords:
        return []
    list_keywords = sorted(list_keywords, key=lambda tup: tup[4], reverse=order)
    if len(list_keywords) > 2:
        list_keywords = list_keywords[:2]
    return list_keywords


def init_headline_generator_v2(headline, all_headlines, all_expressions, model, all_labels, configs, gen_method):
    """
    This function, considering a given headline, applies methods to generate new expressions based on proverbs and
    chosen words (computed by different methods).
    @:param gen_method: id representing the used methodology
    0-Substitution; 1-Analogy; 2-Vector Comparison
    @:param sel_methods: ids representing the chosen methods for selecting the best proverbs and
    the best generated expressions. 0 - WE; 1 - TFIDFVectorizer; 2- BERT
    """

    # ----- CONFIGURATION -----
    nlpyport.load_config()
    # Retrieve data for TFIDF computation -----
    tfidf_words, tfidf_occur, first_sel_amount, sel_method = configs

    info = get_info_methods([gen_method, sel_method])
    print("[IDENTIFIER] ", headline, info)
    all_generated_expressions = []
    append_expression = all_generated_expressions.append

    # Structure of Headline: [([0] word; [1] lemma; [2] PoS tag; [3] Form), .. , ()]
    headline_tokens = headline.lower().translate(str.maketrans('', '', string.punctuation)).split()
    headline_keywords = []
    headline_substitutes = []
    keyword_comparison_vec = []
    accepted_pos = []

    if gen_method == 0:
        headline_dets = [find_label(tok, headline_tokens, all_labels) for tok in headline_tokens]
        headline_substitutes = get_headline_substitutes(headline_dets, model, all_labels, amount_subs=10)
    elif gen_method in [1, 2]:
        all_headlines.insert(0, headline)
        # ----- Use TF-IDF for disambiguity -----
        # headline_tfidf = get_word_tfidf(0, all_headlines)
        # ----- Use lexicon for disambiguity -----
        headline_tfidf = get_word_tfidf_v2(headline, tfidf_words, tfidf_occur)
        headline_keywords = []
        for hk in headline_tfidf:
            if hk[0] in model.vocab:
                hk_det = find_label(hk[0], headline_tokens, all_labels)
                if hk_det != () and check_pos(hk_det[2]):
                    headline_keywords.append(hk_det)
            if len(headline_keywords) >= 2:
                break
        if not headline_keywords or (len(headline_keywords) < 2 and gen_method == 2):
            print("[ERROR] Invalid headline.")
            return headline, "-", "-"
        # print(headline_keywords)
        keyword_comparison_vec = model.wv[headline_keywords[0][0]] - model.wv[headline_keywords[1][0]]
        accepted_pos = [trim_pos(headline_keywords[0][2]), trim_pos(headline_keywords[1][2])]

    # -------- First Selection -----------
    amount = int(first_sel_amount//3)
    selected_proverbs = [prov[1] for prov in get_first_selection(all_expressions, headline, 1,
                                                                 amount=amount, model=model)]
    selected_proverbs.extend(
        [prov[1] for prov in get_first_selection(all_expressions, headline, 2,
                                                 amount=amount, model=model)])
    random_extension = random.sample(all_expressions, k=int(first_sel_amount//2))
    extension_counter = 0
    for counter in range(len(random_extension)):
        if random_extension[counter] not in selected_proverbs:
            selected_proverbs.append(random_extension[counter])
            extension_counter += 1
        if extension_counter >= amount:
            break

    for counter, proverb in enumerate(selected_proverbs):
        tmp_proverb = (proverb.lower(), 0)
        # Structure of keywords: [( [0] word; [1] lemma; [2] PoS tag;  [3] Form ; [4] Similarity), .. ,()]

        # ----- Use TF-IDF for disambiguity -----
        # best_prov_keywords = get_best_keywords(get_word_tfidf(prov_index, all_proverbs), all_labels, model, True)
        # ----- Use lexicon for disambiguity -----
        best_prov_keywords = get_best_keywords(get_word_tfidf_v2(proverb, tfidf_words, tfidf_occur),
                                               all_labels, model, False)

        # ----------------- Substituition Method -----------------
        if gen_method == 0:
            for keyword_det in best_prov_keywords:
                if not keyword_det:
                    continue
                list_substitutes = get_substitutes_v2(keyword_det, model, headline_substitutes, all_labels, amount=5)
                for sub_det in list_substitutes:
                    append_expression(re.sub(r"\b%s\b" % keyword_det[0], sub_det[0], tmp_proverb[0]))

        # ----------------- Analogy Method -----------------
        elif gen_method == 1 and len(best_prov_keywords) >= 2:
            h_keyword1_det = headline_keywords[0]
            if h_keyword1_det != () and len(h_keyword1_det) >= 4:
                hk_pos = trim_pos(h_keyword1_det[2])
                # print("Headline_keyword: ", h_keyword1_det, hk_pos, best_prov_keywords)
                if hk_pos in best_prov_keywords[0][2] and h_keyword1_det[0] != best_prov_keywords[0][0]:
                    tmp_subs = model.wv.most_similar(positive=[h_keyword1_det[0], best_prov_keywords[0][0]],
                                                     negative=[best_prov_keywords[1][0]], topn=10)
                    tmp_subs = [find_label(sub, [sub + ' ' + headline], all_labels) for sub, sim in tmp_subs]
                    tmp_gen = get_generated_expressions(proverb, h_keyword1_det, best_prov_keywords, keyword_id=0,
                                                        all_substitutes=tmp_subs, all_labels=all_labels)
                    for expression in tmp_gen:
                        append_expression(expression)

                if hk_pos in best_prov_keywords[1][2] and h_keyword1_det[0] != best_prov_keywords[1][0]:
                    tmp_subs = model.wv.most_similar(positive=[h_keyword1_det[0], best_prov_keywords[1][0]],
                                                     negative=[best_prov_keywords[0][0]], topn=10)
                    tmp_subs = [find_label(sub, [sub + ' ' + headline], all_labels) for sub, sim in tmp_subs]
                    tmp_gen = get_generated_expressions(proverb, h_keyword1_det, best_prov_keywords, keyword_id=1,
                                                        all_substitutes=tmp_subs, all_labels=all_labels)
                    for expression in tmp_gen:
                        append_expression(expression)

        # ----------------- Vector comparison Method -----------------
        elif gen_method == 2 and len(best_prov_keywords) >= 2:
            prov_tokens = proverb.lower().translate(str.maketrans('', '', string.punctuation)).split()
            prov_key1, prov_key2, sub_order = get_comparison_keywords(keyword_comparison_vec, prov_tokens, accepted_pos,
                                                                      all_labels, model)
            if not prov_key1 or not prov_key2:
                continue
            gen_exp = get_generated_expressions_compvec(proverb, headline_keywords, prov_key1, prov_key2,
                                                        sub_order, all_labels)
            for expression in gen_exp:
                append_expression(expression)

    if not all_generated_expressions:
        print("[ERROR] Could not generate any expressions.")
        return headline, "-", "-"

    # ----- TfidfVectorizer -----
    if sel_method == 1:
        selected_proverbs = init_prov_selector_standard(0, [headline], all_generated_expressions, amount=1)
        while len(selected_proverbs) == 1:
            selected_proverbs = selected_proverbs[0]
        print("SELECTED PROVERBS 1:\t", len(selected_proverbs), selected_proverbs, '\n')
        return selected_proverbs

    # ----- BERT -----
    elif sel_method == 2:
        selected_proverbs = init_prov_selector_bert(7, [headline], all_generated_expressions, amount=1)
        while len(selected_proverbs) == 1:
            selected_proverbs = selected_proverbs[0]
        print("\nSELECTED PROVERBS 2:\t", len(selected_proverbs), selected_proverbs, '\n')
        return selected_proverbs

    else:
        print("[ERROR] Invalid selection method.")
