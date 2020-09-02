import sys
import NLPyPort as nlpyport

from load_config import load_config_selector
from proverb_selector.sel_approach_standard.standard_approach import init_prov_selector_standard
from proverb_selector.sel_approach_transformer.transformer_approach import init_prov_selector_bert
from proverb_selector.sel_approach_we.we_object_manager import *
from proverb_selector.sel_approach_we.we_approach import init_prov_selector_we


def init_prov_selector(sim_alg, prov_location):
    config_files = load_config_selector()
    # Similarity Algorithm is given by the first argument
    proverbs = data_retrieval(prov_location, 'UTF-8')
    # input_text = data_retrieval(inp_location, 'UTF-8')
    input_text = ["A água salobra faz a terra dura e a menina magra. ",
                  "Produção de combustíveis fósseis cresce 50% acima do necessário para travar aquecimento global."]
    print("[INPUT] " + str(input_text))

    if sim_alg in [0, 1]:  # Choosing standard approaches
        print("RETURN: ", init_prov_selector_standard(sim_alg, input_text, proverbs, amount=1))
    elif sim_alg == 2:  # Jaccard
        model_filename = config_files[0]
        init_prov_selector_we(input_text, proverbs, sim_alg, model_filename, amount=1)
    elif sim_alg in [3, 4]: # Choosing Word Embeddings approach with glove (with and without tfidf)
        model_filename = config_files[1]
        init_prov_selector_we(input_text, proverbs, sim_alg, model_filename, amount=1)
    elif sim_alg in [5, 6]:  # Choosing Word Embeddings approach with fasttext (with and without tfidf)
        model_filename = config_files[2]
        init_prov_selector_we(input_text, proverbs, sim_alg, model_filename, amount=1)
    elif sim_alg in [7, 8]:  # Choosing Transformers approach (with and without fine tuning)
        init_prov_selector_bert(sim_alg, input_text, proverbs, amount=1)
    else:
        print("[ERROR] No correct algorithm chosen.")
    return


if __name__ == "__main__":
    # Load configuration for NLPyPort
    nlpyport.load_config()
    # Start logging
    log.basicConfig(format='%(asctime)s - %(message)s', level=log.INFO)
    # news_data_retrieval()
    init_prov_selector(7, 'sel_inputs/proverbios_natura.txt')
