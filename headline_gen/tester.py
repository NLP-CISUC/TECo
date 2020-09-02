import string
import time

from gensim.models import KeyedVectors

import NLPyPort as nlpyport

from proverb_selector.sel_utils.file_manager import read_write_obj_file
from headline_gen.gen_utils.utils_gen import find_label, get_word_tfidf


def get_tfidf(headline, all_words, all_occurrences):
    token_dets = []
    hl_tokens = headline.lower().translate(str.maketrans('', '', string.punctuation)).split()
    for token in hl_tokens:
        if token in all_words:
            token_dets.append((token, all_occurrences[all_words.index(token)]))
    return sorted(token_dets, key=lambda tup: tup[1], reverse=False)


def test_data_creation(tokens, model, filename, all_labels):
    tfidf_words = []
    tfidf_occurrences = []
    nlpyport.load_config()
    with open(filename, 'r') as proverb_file:
        p_reader = proverb_file.readlines()
        # print(len(p_reader))
        for row in p_reader:
            line = row.split()
            occurrences = int(line[0])
            if line[1] in model.vocab:
                for label in all_labels:
                    if label[0] == line[1] and line[1] not in tfidf_words:
                        tfidf_occurrences.append(occurrences)
                        tfidf_words.append(line[1])
                        break

    print(len(tfidf_words), tfidf_words)
    read_write_obj_file(0, tfidf_words, 'gen_inputs/tfidf_words.pk1')
    print(len(tfidf_occurrences), tfidf_occurrences)
    read_write_obj_file(0, tfidf_occurrences, 'gen_inputs/tfidf_occur.pk1')


if __name__ == '__main__':
    """hl_example = "Quando vai acabar a quarentena? Estou completamente farto disto."
    all_labels = read_write_obj_file(1, None, 'gen_inputs/list_labels_v3.pk1')
    tfidf_words = read_write_obj_file(1, None, 'gen_inputs/tfidf_words.pk1')
    tfidf_occur = read_write_obj_file(1, None, 'gen_inputs/tfidf_occur.pk1')
    selected = read_write_obj_file(1, None, 'gen_inputs/selected_rank.pk1')
    all_headlines = [s[0] for s in selected]
    all_headlines.insert(0, hl_example)

    start = time.time()
    test_data_creation(
        hl_example.split(),
        KeyedVectors.load('C:/Disciplinas/Tese/ThesisWork/proverb_selector/sel_inputs/we_models/model_glove300.model'),
        'gen_inputs/formas_cetempublico_utf8.txt',
        all_labels)
    print("NEW ", get_tfidf(hl_example, tfidf_words, tfidf_occur))
    print("OLD ", get_word_tfidf(0, all_headlines))
    end = time.time()
    print("time: ", end-start)
    x = read_write_obj_file(1, None, 'gen_inputs/list_labels_V2.pk1')
    y = read_write_obj_file(1, None, 'gen_inputs/list_labels_v3.pk1')
    print(len(x), len(y))"""
    x = 'afeicão é manual'
    print(x, x.lower().translate(str.maketrans('', '', string.punctuation)).strip())
