import time

import numpy as np
import string

import NLPyPort as nlpyport
from proverb_selector.sel_utils.file_manager import *
from bert_serving.client import BertClient
from bert_serving.server import BertServer
from bert_serving.server.helper import get_args_parser


def init_prov_selector_bert(alg, input_text, proverbs, amount):
    info_sim = 'BERT'
    sim = []
    if alg == 7:
        # bc_client = BertClient(ip='10.3.2.102')
        print("Initializing BERT Client...")
        bc_client = BertClient()
        print("BERT Client initialization successful.")
        # print(input_text)
        inp_matrix = [bc_client.encode([i]) for i in input_text]
        inp_vectors = [get_sentence_vector_bert(i_v[0]) for i_v in inp_matrix]
        try:
            prov_vectors = read_write_obj_file(1, None, 'models_db/bert_pretrained_models/prov_vec.pk1')
        except FileNotFoundError:
            print("[ERROR] Encoding BERT vectors now...")
            prov_matrix = [bc_client.encode([p]) for p in proverbs]
            prov_vectors = [get_sentence_vector_bert(p_v[0]) for p_v in prov_matrix]
            read_write_obj_file(0, prov_vectors, 'models_db/bert_pretrained_models/prov_vec.pk1')

        sim = [[] for i in range(len(input_text))]

        for counter_inp in range(len(input_text)):
            for counter_prov in range(len(proverbs)):
                sim[counter_inp].append(np.dot(inp_vectors[counter_inp], prov_vectors[counter_prov]) / (
                        np.linalg.norm(inp_vectors[counter_inp]) * np.linalg.norm(prov_vectors[counter_prov])))

        # BertServer.shutdown(args=args)
        bc_client.close()
    return selector(input_text, proverbs, sim, info_sim, amount=amount)


def preprocess_bert(input_text):
    txt = [nlpyport.tokenize_from_string(i.lower()) for i in input_text]
    punc = string.punctuation + '\''
    for t in txt:
        tmp_t = t.copy()
        for token in tmp_t:
            if token in punc or token == '\n':
                t.remove(token)
    return txt


def get_sentence_vector_bert(sentence_vector):
    tmp = np.delete(sentence_vector, 0, axis=0)  # delete CLS vector
    tmp = np.delete(tmp, -1, axis=0)  # delete SEP vector

    return np.mean(tmp, axis=0, dtype=np.float64)


if __name__ == '__main__':
    start = time.time()
    print(init_prov_selector_bert(7, ["Exemplo de texto\n"],
                                  read_write_obj_file(1, None, '../../models_db/prov_movies.pk1'), 1))
    print("EXEC_TIME: ", time.time()-start)
