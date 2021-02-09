import string
import random

from proverb_selector.sel_approach_standard.standard_approach import init_prov_selector_standard
from proverb_selector.sel_approach_transformer.transformer_approach import init_prov_selector_bert_service
from proverb_selector.sel_approach_transformer.transformer_approach import init_prov_selector_bert
from headline_gen.gen_utils.utils_gen import get_sentence_vector

TFIDF="TFIDF"
WE="WE"
BERT="BERT"

def get_first_selection(all_proverbs, headline, sel_method, amount, model):
    selected_proverbs = []
    # Word Embeddings
    if sel_method == WE:
        headline_tokens = headline.lower().translate(str.maketrans('', '', string.punctuation)).split()
        headline_vec = get_sentence_vector(headline_tokens, model)
        for prov in all_proverbs:
            prov_tokens = prov.lower().translate(str.maketrans('', '', string.punctuation)).split()
            prov_vec = get_sentence_vector(prov_tokens, model)
            if not any(prov_vec):
                continue
            cos = model.wv.cosine_similarities(headline_vec, [prov_vec])[0]
            selected_proverbs.append((headline, prov, cos))
        selected_proverbs = random.shuffle(selected_proverbs) if selected_proverbs[0][2] == selected_proverbs[-1][2] else sorted(selected_proverbs, key=lambda tup: tup[2], reverse=True)[:amount]
        return selected_proverbs

    # TFIDFVectorizer
    elif sel_method == TFIDF:
        selected_proverbs = init_prov_selector_standard(0, headline, all_proverbs, amount=amount)
        # print("\nSELECTED PROVERBS:\t", len(selected_proverbs), selected_proverbs)
        return selected_proverbs

    # BERT
    elif sel_method == BERT:
        selected_proverbs = init_prov_selector_bert_service(headline, all_proverbs, amount=amount)
        #selected_proverbs = init_prov_selector_bert(7, [headline], all_proverbs, amount=amount)
        # print("\nSELECTED PROVERBS:\t", len(selected_proverbs), selected_proverbs)
        return selected_proverbs

    return selected_proverbs
