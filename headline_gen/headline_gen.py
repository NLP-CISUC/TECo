"""
@author: Rui Mendes, Hugo Gonçalo Oliveira
"""
import random
import re

from proverb_selector.sel_approach_standard.standard_approach import init_prov_selector_standard
from proverb_selector.sel_approach_transformer.transformer_approach import init_prov_selector_bert
from proverb_selector.sel_approach_we.we_approach import init_prov_selector_we
from gen_methods.selection_methods import *
from gen_methods.substitution_methods import *
from teco_twitterbot.twitter_utils.twitter_manager import get_info_methods
from gen_utils.utils_gen import *

SUBSTITUTION="Subs"
ANALOGY="Analogy"
VEC_DIFF="VecDiff"

def get_best_keywords(sentence_words, all_labels, model, order):
    """
    This function chooses the most important words from a sentence from their Tf-Idf value.
    @:param sentence_words: list of tuples, each containing a word from the sentence and its Tf-Idf value
    @:param order: if True: descending order, if False: ascending order
    @:returns list of tuples with the structures representing the two most valuable keywords.
    """
    list_keywords = []
    sentence_tokens = [w[0] for w in sentence_words]
    for word, similarity in sentence_words:
        word_det = find_label(word, sentence_tokens, all_labels)
        if word_det == () or not check_pos(word_det[2]) or word_det[0] not in model.vocab:
            continue
        list_keywords.append((word_det[0], word_det[1], word_det[2], word_det[3], similarity))

    if not list_keywords:
        return []
    list_keywords = sorted(list_keywords, key=lambda tup: tup[4], reverse=order)

    #print(sentence_words, list_keywords)

    if len(list_keywords) > 2:
        list_keywords = list_keywords[:2]
    return list_keywords


def init_headline_generator_v2(headline, headline_tokens, use_expressions, model, dict_forms_labels, dict_lemmas_labels, tfidf, gen_method):
    """
    This function, considering a given headline, applies methods to generate new expressions based on proverbs and
    chosen words (computed by different methods).
    @:param gen_method: id representing the used methodology
    """

    # ----- CONFIGURATION -----
    nlpyport.load_config()
    # Retrieve data for TFIDF computation -----

    #info = get_info_methods([gen_method, sel_method])
    print("[IDENTIFIER] ", headline, gen_method)
    all_generated_expressions = []

    # -------- Adaptation -----------
    if gen_method == SUBSTITUTION:
        all_generated_expressions = substitution_many(use_expressions, headline_tokens, tfidf, dict_forms_labels, dict_lemmas_labels, model)
    elif gen_method == ANALOGY:
        all_generated_expressions = analogy_many(use_expressions, headline, headline_tokens, tfidf, dict_forms_labels, dict_lemmas_labels, model)
    elif gen_method == VEC_DIFF:
        all_generated_expressions = vecdiff_many(use_expressions, headline, headline_tokens, tfidf, dict_forms_labels, dict_lemmas_labels, model)

    if not all_generated_expressions:
        print("[ERROR] Could not generate expression with "+gen_method)
        return None

    return all_generated_expressions


def substitution_many(proverbs, headline_tokens, tfidf, dict_forms_labels, dict_lemmas_labels, model):
    headline_dets = [find_label(tok, headline_tokens, dict_forms_labels) for tok in headline_tokens]
    headline_substitutes = get_headline_substitutes(headline_dets, model, dict_forms_labels, amount_subs=6)

    all_generated = []
    for proverb in proverbs:
        #tmp_proverb = (proverb.lower(), 0)

        best_prov_keywords = get_best_keywords(get_word_tfidf_v2(proverb, tfidf), dict_forms_labels, model, False)
        generated = substitution_one(proverb, best_prov_keywords, headline_substitutes, dict_lemmas_labels, model)
        if generated:
            all_generated.extend(generated)
    return all_generated


def substitution_one(proverb, best_prov_keywords, headline_substitutes, dict_lemmas_labels, model):
    generated_expressions = []
    for keyword_det in best_prov_keywords:
        if not keyword_det:
            continue
        list_substitutes = get_substitutes_v2(keyword_det, model, headline_substitutes, dict_lemmas_labels, amount=3)
        for sub_det in list_substitutes:
            gen = re.sub(r"\b%s\b" % keyword_det[0], sub_det[0], proverb)
            #print("Generated with "+SUBSTITUTION+": " + proverb.strip() + " -> " + gen.strip())
            generated_expressions.append(gen)

    return generated_expressions


def pre_analogy_vecdiff(headline, headline_tokens, tfidf, all_labels, model):
    #all_headlines.insert(0, headline)
    # ----- Use TF-IDF for disambiguity -----
    # headline_tfidf = get_word_tfidf(0, all_headlines)
    # ----- Use lexicon for disambiguity -----
    headline_tfidf = get_word_tfidf_v2(headline, tfidf, input_tokens=headline_tokens)

    headline_keywords = []
    for hk in headline_tfidf:
        if hk[0] in model.vocab:
            hk_det = find_label(hk[0], headline_tokens, all_labels)
            if hk_det and check_pos(hk_det[2]) and not(aux_verb(hk_det)):
                    headline_keywords.append(hk_det)
        if len(headline_keywords) >= 2:
            break
    print("[Keywords]", headline_keywords)
    if not headline_keywords or len(headline_keywords) < 2:
        print("[ERROR] Invalid headline.")
        return None, None, None, None
    keyword_comparison_vec = model.wv[headline_keywords[0][0]] - model.wv[headline_keywords[1][0]]
    accepted_pos = [trim_pos(headline_keywords[0][2]), trim_pos(headline_keywords[1][2])]
    return headline_keywords, headline_tfidf, keyword_comparison_vec, accepted_pos


def analogy_many(proverbs, headline, headline_tokens, tfidf, dict_forms_labels, dict_lemmas_labels, model):
    headline_keywords, headline_tfidf, keyword_comparison_vec, accepted_pos = pre_analogy_vecdiff(headline, headline_tokens, tfidf, dict_forms_labels, model)
    if not headline_keywords:
        return None

    all_generated = []
    for proverb in proverbs:
        #tmp_proverb = (proverb.lower(), 0)
        # Structure of keywords: [( [0] word; [1] lemma; [2] PoS tag;  [3] Form ; [4] Similarity), .. ,()]

        # ----- Use TF-IDF for disambiguity -----
        # best_prov_keywords = get_best_keywords(get_word_tfidf(prov_index, all_proverbs), all_labels, model, True)
        # ----- Use lexicon for disambiguity -----
        best_prov_keywords = get_best_keywords(get_word_tfidf_v2(proverb, tfidf), dict_forms_labels, model, False)

        if len(best_prov_keywords) >= 2:
            generated = analogy_one(proverb, best_prov_keywords, headline, headline_keywords, dict_forms_labels, dict_lemmas_labels, model, min_sim=0.5)
            if generated:
                all_generated.extend(generated)
    return all_generated


def analogy_one(proverb, best_prov_keywords, headline, headline_keywords, dict_forms_labels, dict_lemmas_labels, model, min_sim=0.5):
    generated_expressions = []
    h_keyword1_det = headline_keywords[0]
    if h_keyword1_det != () and len(h_keyword1_det) >= 4:
        hk_pos = trim_pos(h_keyword1_det[2])
        # print("Headline_keyword: ", h_keyword1_det, hk_pos, best_prov_keywords)

        # TODO: organizar...
        if hk_pos in best_prov_keywords[0][2] and h_keyword1_det[0] != best_prov_keywords[0][0]:
            tmp_subs = model.wv.most_similar(positive=[h_keyword1_det[0], best_prov_keywords[0][0]],
                                             negative=[best_prov_keywords[1][0]], topn=5)

            filt_subs = []
            for sub, sim in tmp_subs:
                if sim >= min_sim:
                    filt_subs.append(sub)

            label_subs = [find_label(sub, [sub + ' ' + headline], dict_forms_labels) for sub in filt_subs]
            tmp_gen = get_generated_expressions(proverb, h_keyword1_det, best_prov_keywords, keyword_id=0,
                                                all_substitutes=label_subs, all_labels=dict_lemmas_labels)
            for expression in tmp_gen:
                #print("Generated with "+ANALOGY+": " + proverb.strip() + " -> " + expression.strip())
                generated_expressions.append(expression)

        if hk_pos in best_prov_keywords[1][2] and h_keyword1_det[0] != best_prov_keywords[1][0]:
            tmp_subs = model.wv.most_similar(positive=[h_keyword1_det[0], best_prov_keywords[1][0]],
                                             negative=[best_prov_keywords[0][0]], topn=5)

            filt_subs = []
            for sub, sim in tmp_subs:
                if sim >= min_sim:
                    filt_subs.append(sub)

            label_subs = [find_label(sub, [sub + ' ' + headline], dict_forms_labels) for sub in filt_subs]
            tmp_gen = get_generated_expressions(proverb, h_keyword1_det, best_prov_keywords, keyword_id=1,
                                                all_substitutes=label_subs, all_labels=dict_lemmas_labels)
            for expression in tmp_gen:
                #print("Generated with "+ANALOGY+": " + proverb.strip() + " -> " + expression.strip())
                generated_expressions.append(expression)

    return generated_expressions


def vecdiff_many(proverbs, headline, headline_tokens, tfidf, dict_forms_labels, dict_lemmas_labels, model):
    headline_keywords, headline_tfidf, keyword_comparison_vec, accepted_pos = pre_analogy_vecdiff(headline,
                                                                                                  headline_tokens,
                                                                                                  tfidf,
                                                                                                  dict_forms_labels, model)
    if not headline_keywords:
        return None

    all_generated = []
    for counter, proverb in enumerate(proverbs):
        tmp_proverb = (proverb.lower(), 0)
        # Structure of keywords: [( [0] word; [1] lemma; [2] PoS tag;  [3] Form ; [4] Similarity), .. ,()]

        generated = vecdiff_one(proverb, headline_keywords, accepted_pos, keyword_comparison_vec, dict_forms_labels, dict_lemmas_labels, model)
        if generated:
            all_generated.extend(generated)

    return all_generated


def vecdiff_one(proverb, headline_keywords, accepted_pos, keyword_comparison_vec, dict_forms_labels, dict_lemmas_labels, model):
    generated_expressions = []

    prov_tokens = proverb.lower().translate(str.maketrans('', '', string.punctuation)).split()
    prov_key1, prov_key2, sim, sub_order = get_comparison_keywords(keyword_comparison_vec, prov_tokens, accepted_pos,
                                                              dict_forms_labels, model, min_sim=0.1)
    if not prov_key1 or not prov_key2:
        return None

    gen_exp = get_generated_expressions_vecdiff(proverb, headline_keywords, prov_key1, prov_key2,
                                                sub_order, dict_lemmas_labels)

    for i, expression in enumerate(gen_exp):
        #print("Generated with "+VEC_DIFF+": " + proverb.strip() + " -> " + expression.strip(), "** ", sim[0])
        generated_expressions.append(expression)

    return generated_expressions