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

#REGEX_QUOTATION="«|»"


def get_best_keywords(sentence_words, all_labels, model, reverse_order):
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
    list_keywords = sorted(list_keywords, key=lambda tup: tup[4], reverse=reverse_order)

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

    #TODO: se expressão tiver menos de 5 tokens, usar sempre Subs?

    # -------- Adaptation -----------
    if gen_method == SUBSTITUTION:
        all_generated_expressions = substitution_many(headline, use_expressions, headline_tokens, tfidf, dict_forms_labels, dict_lemmas_labels, model)
    elif gen_method == ANALOGY:
        all_generated_expressions = analogy_many(use_expressions, headline, headline_tokens, tfidf, dict_forms_labels, dict_lemmas_labels, model)
    elif gen_method == VEC_DIFF:
        all_generated_expressions = vecdiff_many(use_expressions, headline, headline_tokens, tfidf, dict_forms_labels, dict_lemmas_labels, model)

    if not all_generated_expressions:
        print("[ERROR] Could not generate expression with "+gen_method)
        return None

    return all_generated_expressions


def substitution_many(headline, proverbs, headline_tokens, tfidf, dict_forms_labels, dict_lemmas_labels, model):
    headline_keywords = get_headline_keywords(headline, headline_tokens, tfidf, dict_forms_labels, model, min=1, max=5)
    headline_substitutes = get_headline_substitutes(headline_keywords, model, dict_forms_labels, top_similar=5)
    print("[Sub Candidates]", headline_substitutes)

    all_generated = []
    for proverb in proverbs:
        #tmp_proverb = (proverb.lower(), 0)

        best_prov_keywords = get_best_keywords(get_words_relevance(proverb, tfidf), dict_forms_labels, model, False)
        generated = substitution_one(proverb, best_prov_keywords, headline_substitutes, dict_lemmas_labels, model)
        if generated:
            all_generated.extend(generated)
    return all_generated


def substitution_one(proverb, best_prov_keywords, headline_substitutes, dict_lemmas_labels, model):
    generated_expressions = []
    for keyword_det in best_prov_keywords:
        if not keyword_det:
            continue
        list_substitutes = get_substitutes_v2(keyword_det, model, headline_substitutes, dict_lemmas_labels, amount=5)
        for sub_det in list_substitutes:
            gen = re.sub(r"\b%s\b" % keyword_det[0], sub_det[0], proverb)
            #print("Generated with "+SUBSTITUTION+": " + proverb.strip() + " -> " + gen.strip())
            generated_expressions.append(gen)

    return generated_expressions

def get_headline_keywords(headline, headline_tokens, tfidf, all_labels, model, min=2, max=2):
    headline_tfidf = get_words_relevance(headline, tfidf, ascending_df=True, input_tokens=headline_tokens)
    #print("TFIDF", headline_tfidf)
    headline_keywords = []
    for hk in headline_tfidf:
        if hk[0] in model.vocab:
            hk_det = find_label(hk[0], headline_tokens, all_labels)
            if hk_det and check_pos(hk_det[2]) and not(aux_verb(hk_det)):
                headline_keywords.append(hk_det)
        if len(headline_keywords) >= max:
            break

    if len(headline_keywords) < min:
        print("[INFO] Not enough keywords in headline.")
        return None
    else:
        print("[Keywords]", headline_keywords)
    return headline_keywords

def analogy_many(proverbs, headline, headline_tokens, tfidf, dict_forms_labels, dict_lemmas_labels, model):
    headline_keywords = get_headline_keywords(headline, headline_tokens, tfidf, dict_forms_labels, model, min=2, max=3)
    if not headline_keywords:
        return None

    all_generated = []
    for proverb in proverbs:
        # Structure of keywords: [( [0] word; [1] lemma; [2] PoS tag;  [3] Form ; [4] Similarity), .. ,()]
        best_prov_keywords = get_best_keywords(get_words_relevance(proverb, tfidf), dict_forms_labels, model, False)

        #needs at least two keywords to replace!
        if len(best_prov_keywords) >= 2:
            generated = analogy_one(proverb, best_prov_keywords, headline, headline_keywords, dict_forms_labels, dict_lemmas_labels, model, min_sim=0.5)
            if generated:
                all_generated.extend(generated)
    return all_generated

'''
def analogy_one(proverb, prov_keywords, headline, headline_keywords, dict_forms_labels, dict_lemmas_labels, model, min_sim=0.4):
    generated_expressions = []
    for h_keyword in headline_keywords:

        if h_keyword and len(h_keyword) >= 4:
            hk_pos = trim_pos(h_keyword[2])
            # print("Headline_keyword: ", h_keyword1_det, hk_pos, best_prov_keywords)

            # TODO: organizar...
            if hk_pos in prov_keywords[0][2] and h_keyword[0] != prov_keywords[0][0]:
                tmp_subs = model.wv.most_similar(positive=[h_keyword[0], prov_keywords[0][0]],
                                                 negative=[prov_keywords[1][0]], topn=5)
            filt_subs = []
            for sub, sim in tmp_subs:
                if sim >= min_sim:
                    filt_subs.append(sub)

            label_subs = [find_label(sub, [sub + ' ' + headline], dict_forms_labels) for sub in filt_subs]
            tmp_gen = get_generated_expressions(proverb, h_keyword, prov_keywords, keyword_id=0,
                                                all_substitutes=label_subs, all_labels=dict_lemmas_labels)
            for expression in tmp_gen:
                #print("Generated with "+ANALOGY+": " + proverb.strip() + " -> " + expression.strip())
                generated_expressions.append(expression)

            if hk_pos in prov_keywords[1][2] and h_keyword[0] != prov_keywords[1][0]:
                tmp_subs = model.wv.most_similar(positive=[h_keyword[0], prov_keywords[1][0]],
                                                 negative=[prov_keywords[0][0]], topn=5)

            filt_subs = []
            for sub, sim in tmp_subs:
                if sim >= min_sim:
                    filt_subs.append(sub)

            label_subs = [find_label(sub, [sub + ' ' + headline], dict_forms_labels) for sub in filt_subs]
            tmp_gen = get_generated_expressions(proverb, h_keyword, prov_keywords, keyword_id=1,
                                                all_substitutes=label_subs, all_labels=dict_lemmas_labels)
            for expression in tmp_gen:
                #print("Generated with "+ANALOGY+": " + proverb.strip() + " -> " + expression.strip())
                generated_expressions.append(expression)

    return generated_expressions
'''

def analogy_one(expression, exp_keywords, headline, headline_keywords, dict_forms_labels, dict_lemmas_labels, model,
                   min_sim=0.45):

    #print("expression=", expression, "|| exp_keywords=", exp_keywords)
    #print("headline=", headline, "|| headline_keywords=", headline_keywords)

    generated_expressions = []
    for h_keyword in headline_keywords:
        if h_keyword and len(h_keyword) >= 4:
            hk_pos = trim_pos(h_keyword[2])
            # print("Headline_keyword: ", h_keyword1_det, hk_pos, best_prov_keywords)

            for i_exp in range(len(exp_keywords)):
                if hk_pos in exp_keywords[i_exp][2] and h_keyword[0] != exp_keywords[i_exp][0]:
                    for j_exp in range(len(exp_keywords)):

                        if i_exp == j_exp:
                            continue

                        tmp_subs = model.wv.most_similar(positive=[h_keyword[0], exp_keywords[i_exp][0]],
                                                         negative=[exp_keywords[j_exp][0]], topn=5)

                        #print(h_keyword, exp_keywords[i_exp], exp_keywords[j_exp], tmp_subs)

                        filt_subs = [] #quadruplos: k1->sub_k1, k2->sub_k2
                        for sub, sim in tmp_subs:
                            if sim >= min_sim:
                                #print("**", (exp_keywords[i_exp], h_keyword, exp_keywords[j_exp], sub))
                                #even if h_keyword and sub are different, they can have the same lemma and, after inflection, become the same
                                filt_subs.append((exp_keywords[i_exp], h_keyword, exp_keywords[j_exp], sub))

                        if filt_subs:
                            gen_expressions = generate_analog_expressions(headline, expression, filt_subs, dict_forms_labels, dict_lemmas_labels)
                            if gen_expressions:
                                #print(filt_subs, "-->", gen_expressions)
                                generated_expressions.extend(gen_expressions)

    return generated_expressions



def vecdiff_many(proverbs, headline, headline_tokens, tfidf, dict_forms_labels, dict_lemmas_labels, model):
    headline_keywords = get_headline_keywords(headline, headline_tokens, tfidf, dict_forms_labels, model, min=2, max=2)

    if not headline_keywords or len(headline_keywords) < 2:
        return None

    keyword_vec_diff = model.wv[headline_keywords[0][0]] - model.wv[headline_keywords[1][0]]
    accepted_pos = [trim_pos(headline_keywords[0][2]), trim_pos(headline_keywords[1][2])]

    all_generated = []
    generate_with_subs = []
    for counter, proverb in enumerate(proverbs):
        # Structure of keywords: [( [0] word; [1] lemma; [2] PoS tag;  [3] Form ; [4] Similarity), .. ,()]

        generated, use_subs = vecdiff_one(proverb, headline_keywords, accepted_pos, keyword_vec_diff, dict_forms_labels, dict_lemmas_labels, model)
        if generated:
            all_generated.extend(generated)

    # if not enough tokens, will generate with subs...
    '''
        elif use_subs:
            generate_with_subs.append(proverb)
    
    if generate_with_subs:
        with_subs = substitution_many(headline, generate_with_subs, headline_tokens, tfidf, dict_forms_labels, dict_lemmas_labels, model)
        if with_subs:
            all_generated.extend(with_subs)
    '''

    return all_generated


def vecdiff_one(proverb, headline_keywords, accepted_pos, keyword_comparison_vec, dict_forms_labels, dict_lemmas_labels, model):
    generated_expressions = []

    prov_tokens = get_tokens(proverb)

    #If not enough tokens, original expression becomes harder to identify
    if len(prov_tokens) < 5:
        return None, True

    prov_key1, prov_key2, sim = get_comparable_keywords(keyword_comparison_vec, prov_tokens, accepted_pos, dict_forms_labels, model, min_sim=0.1)
    if not prov_key1 or not prov_key2:
        return None, False

    #print("Sim=", sim, headline_keywords[0], headline_keywords[1], prov_key1, prov_key2)

    gen_exp = get_generated_expressions_vecdiff(proverb, headline_keywords, prov_key1, prov_key2, dict_lemmas_labels)

    for i, expression in enumerate(gen_exp):
        #print("Generated with "+VEC_DIFF+": " + proverb.strip() + " -> " + expression.strip(), "** ", sim[0])
        generated_expressions.append(expression)

    return generated_expressions, False