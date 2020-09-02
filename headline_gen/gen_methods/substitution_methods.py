import numpy as np
from gen_utils.syllable_pt import Syllables
from gen_utils.utils_gen import *


# ------------------- Substitution Method -----------------
def get_headline_substitutes(headline_dets, model, all_labels, amount_subs):
    final_words = []
    append_final_list = final_words.append
    final_score = []
    append_final_score = final_score.append

    for hl_keyword in headline_dets:
        if hl_keyword != () and hl_keyword[0] in model.vocab:
            tmp_list = model.wv.most_similar(positive=hl_keyword[0], topn=amount_subs)
            # print("---", tmp_list)
            # append_all_possibilities([word[0] for word in tmp])
            for word_position, word_det in enumerate(tmp_list):
                # print("word_det ", word_det, final_words, final_score)
                word_index = find_index(word_det[0], final_words)

                if word_index != -1:
                    final_score[word_index] += len(tmp_list) - word_position  # Score based on the position
                    final_score[word_index] += 1  # Score based on occurrences
                    continue

                tmp = find_label(word_det[0], [word_det[0]], all_labels)
                if tmp != ():
                    append_final_list(tmp)
                    append_final_score(1)  # occurrence
                    final_score[-1] += len(tmp_list) - word_position

    final_list = zip(final_words, final_score)
    return sorted(final_list, key=lambda tup: tup[1], reverse=True)


def get_substitutes_v2(keyword_det, model, all_substitutes, all_labels, amount):
    """
    This function defines the list of the best substitutes for the given keyword, considering its Pos and form, while
    being semantically similar to any of the headline's keywords.
    @:param keyword_det: structure of a selected proverb's keyword
    @:param model: word2vec model
    @:param all_substitutes: array composed by structures of each of the headline's keywords
    @:returns list of substitutes: [ ( (headline_og_word, lemma, PoS, form, similarity)
                                       [(sub0_word, sub0_lemma, sub0_PoS, sub0_form, sub0_sim), .., ()]),
                                     (), .. , () ]
    """
    # print(keyword_det)
    keyword = keyword_det[0]
    keyword_pos = trim_pos(keyword_det[2])
    keyword_vec = model.wv[keyword]
    # keyword_syllables = len(Syllables(keyword).make_division())
    list_possibilities = []
    append_possibilities = list_possibilities.append

    for substitute in all_substitutes:
        substitute_det = substitute[0]
        # if substitute_det != () and keyword_pos in substitute_det[2] and \
        #         len(Syllables(substitute_det[0]).make_division()) == keyword_syllables:
        if substitute_det != () and keyword_pos in substitute_det[2]:
            right_form = get_right_form(keyword_det, substitute_det, all_labels)
            if substitute_det[0] == keyword or right_form == keyword:
                continue
            if "######" not in right_form and right_form in model.vocab:
                append_possibilities((right_form, model.wv.cosine_similarities(keyword_vec, [model.wv[right_form]])[0]))
            elif "######" not in right_form and right_form not in model.vocab:
                append_possibilities(
                    (substitute_det[0], model.wv.cosine_similarities(keyword_vec, [model.wv[substitute_det[0]]])[0]))

    return sorted(list_possibilities, key=lambda tup: tup[1], reverse=True)[:amount]


# ------------------- Analogy Method -----------------
def get_generated_expressions(proverb, headline_keyword, prov_keywords, keyword_id, all_substitutes, all_labels):
    positive_keyword = prov_keywords[keyword_id]
    negative_keyword = prov_keywords[keyword_id - 1]
    # keyword_syllables = len(Syllables(positive_keyword[0]).make_division())
    prov_tokens = proverb.lower().translate(str.maketrans('', '', string.punctuation)).split()
    # print("prov_tokens ", prov_tokens)
    tmp_hk_rightform = get_right_form(positive_keyword, headline_keyword, all_labels)
    tmp_proverb = proverb
    generated_expressions = []
    append_gen_expression = generated_expressions.append
    # append_gen_expression(proverb)
    # print("-----------------", proverb)
    # print(headline_keyword, positive_keyword, negative_keyword)

    if trim_pos(headline_keyword[2]) in positive_keyword[2] and tmp_hk_rightform not in prov_tokens \
            and tmp_hk_rightform != '######':
        tmp_proverb = proverb.replace(positive_keyword[0], tmp_hk_rightform)
        prov_tokens = tmp_proverb.lower().translate(str.maketrans('', '', string.punctuation)).split()
        append_gen_expression(tmp_proverb)
        # print("teste1 -> ", tmp_proverb)

        for substitute_det in all_substitutes:
            if substitute_det == () or trim_pos(negative_keyword[2]) not in substitute_det[2] \
                    or negative_keyword[0] == substitute_det[0]:
                continue
            # substitute_syllables = len(Syllables(substitute_det[0]).make_division())
            tmp_hk_rightform = get_right_form(negative_keyword, substitute_det, all_labels)
            # if substitute_syllables == keyword_syllables and tmp_hk_rightform not in prov_tokens:
            if tmp_hk_rightform not in prov_tokens and tmp_hk_rightform != '######':
                tmp_proverb_2 = tmp_proverb.replace(negative_keyword[0], tmp_hk_rightform)
                if tmp_proverb_2 not in generated_expressions:
                    append_gen_expression(tmp_proverb_2)
                    # print("teste2 -> ", tmp_proverb_2)

    return generated_expressions


# ------------------- Vector comparison Method -----------------
def get_comparison_keywords(comparison_vec, prov_tokens, accepted_pos, all_labels, model):
    # print(str(prov_tokens))
    prov_dets = [find_label(tok, prov_tokens, all_labels) for tok in prov_tokens]
    prov_len = len(prov_dets)
    # print(prov_len, prov_dets)
    chosen_pair = (-1, -1, 0)
    for counter1, token1_det in enumerate(prov_dets):
        if token1_det == ():
            continue
        elif counter1 > round(prov_len/2):
            break
        sub_order = -1
        if accepted_pos[0][0] == accepted_pos[1][0] and trim_pos(token1_det[2]) in accepted_pos[0]:
            sub_order = 2
        elif trim_pos(token1_det[2]) in accepted_pos[0]:
            sub_order = 0
        elif trim_pos(token1_det[2]) in accepted_pos[1]:
            sub_order = 1

        # print('token1_det:\t', token1_det)
        if token1_det[0] in model.vocab and sub_order != -1:
            for counter2 in range(counter1+1, prov_len):
                if prov_dets[counter2] == ():
                    continue
                # print(counter1, counter2)
                token2_det = prov_dets[counter2]
                token2_pos = trim_pos(token2_det[2])
                if token2_det[0] not in model.vocab or not check_pos(trim_pos(token2_pos)):
                    continue
                # To ensure the correct order and PoS in the substitution
                if sub_order == 2 and token2_pos != accepted_pos[0]:
                    continue
                elif sub_order == 0 and token2_pos != accepted_pos[1]:
                    continue
                elif sub_order == 1 and token2_pos != accepted_pos[0]:
                    continue
                # print(counter1, token, counter2, prov_tokens[counter2])
                comparison_vec_2 = model.wv[token1_det[0]] - model.wv[token2_det[0]]
                tmp_sim = model.wv.wv.cosine_similarities(np.asarray(comparison_vec), [np.asarray(comparison_vec_2)])
                # dist = np.linalg.norm(a - b)
                if tmp_sim > chosen_pair[2]:
                    chosen_pair = (counter1, counter2, tmp_sim, sub_order)

    if chosen_pair == (-1, -1, 0):
        return None, None, None
    return prov_dets[chosen_pair[0]], prov_dets[chosen_pair[1]], chosen_pair[3]


def get_generated_expressions_compvec(proverb, headline_keywords, prov_key1, prov_key2, sub_order, all_labels):
    selected_expressions = []
    append_expression = selected_expressions.append

    if sub_order in [0, 2]:
        tmp_prov = proverb.replace(prov_key1[0], get_right_form(prov_key1, headline_keywords[0], all_labels))
        # print("0: ", tmp_prov, proverb)
        if "######" not in tmp_prov:
            tmp_2 = tmp_prov.replace(prov_key2[0], get_right_form(prov_key2, headline_keywords[1], all_labels))
            if "######" not in tmp_2:
                # print("0_1: ", tmp_2, proverb)
                append_expression(tmp_2)
            # else:
                # append_expression(tmp_prov)
        else:
            tmp_2 = proverb.replace(prov_key2[0], get_right_form(prov_key2, headline_keywords[1], all_labels))
            if "######" not in tmp_2:
                # print("0_2: ", tmp_2, proverb)
                append_expression(tmp_2)

    if sub_order in [1, 2]:
        tmp_prov = proverb.replace(prov_key1[0], get_right_form(prov_key1, headline_keywords[1], all_labels))
        # print("1: ", tmp_prov, proverb)
        if "######" not in tmp_prov:
            tmp_2 = tmp_prov.replace(prov_key2[0], get_right_form(prov_key2, headline_keywords[0], all_labels))
            if "######" not in tmp_2:
                # print("1_1: ", tmp_2, proverb)
                append_expression(tmp_2)
            # else:
                # append_expression(tmp_prov)
        else:
            tmp_prov = proverb.replace(prov_key1[0], get_right_form(prov_key2, headline_keywords[1], all_labels))
            if "######" not in tmp_prov:
                # print("1_2: ", tmp_prov, proverb)
                append_expression(tmp_prov)

    return selected_expressions
