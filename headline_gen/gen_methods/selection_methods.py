import string

from proverb_selector.sel_approach_standard.standard_approach import init_prov_selector_standard
from proverb_selector.sel_approach_transformer.transformer_approach import init_prov_selector_bert
from gen_utils.utils_gen import get_sentence_vector


def get_first_selection(all_proverbs, headline, sel_method, amount, model):
    headline_vec = get_sentence_vector(headline.lower().translate(str.maketrans('', '', string.punctuation)).split(),
                                       model)
    selected_proverbs = []
    append_proverb = selected_proverbs.append
    # Word Embeddings
    if sel_method == 0:
        for prov in all_proverbs:
            prov_tokens = prov.lower().translate(str.maketrans('', '', string.punctuation)).split()
            prov_vec = get_sentence_vector(prov_tokens, model)
            if not any(prov_vec):
                continue
            append_proverb((headline, prov, model.wv.cosine_similarities(headline_vec, [prov_vec])))
        selected_proverbs = sorted(selected_proverbs, key=lambda tup: tup[1], reverse=True)[:amount]
        # print("\nSELECTED PROVERBS:\t", len(selected_proverbs), selected_proverbs)
        return selected_proverbs

    # TFIDFVectorizer
    elif sel_method == 1:
        selected_proverbs = init_prov_selector_standard(0, [headline], all_proverbs, amount=amount)
        # print("\nSELECTED PROVERBS:\t", len(selected_proverbs), selected_proverbs)
        return selected_proverbs

    # BERT
    elif sel_method == 2:
        selected_proverbs = init_prov_selector_bert(7, [headline], all_proverbs, amount=amount)
        # print("\nSELECTED PROVERBS:\t", len(selected_proverbs), selected_proverbs)
        return selected_proverbs
    return []
