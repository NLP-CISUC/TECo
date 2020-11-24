import string
import time

from gensim.models import KeyedVectors

import NLPyPort as nlpyport

from proverb_selector.sel_utils.file_manager import read_write_obj_file
#from headline_gen.gen_utils.utils_gen import find_label, get_word_tfidf
#from gen_methods.movie_titles import check_movie_pt
from gen_utils.utils_gen import check_pos, trim_pos


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


def create_freq_dict(filename, model, all_labels, min_freq=100):
    dicio = {}
    with open(filename, 'r') as file:
        p_reader = file.readlines()
        for row in p_reader:
            line = row.split()
            occ = int(line[0])
            if occ >= min_freq and line[1] in all_labels and line[1] in model.vocab:
                if line[1] not in dicio:
                    dicio[line[1]] = occ
    read_write_obj_file(0, dicio, '../models_db/dict_freq.pk1')

def labels_2_dict(all_labels):
    dicio = {}
    for l in all_labels:
        if l[0] not in dicio:
            dicio[l[0]] = []
        dicio[l[0]].append(l)
    read_write_obj_file(0, dicio, '../models_db/dict_labels.pk1')

def best_rated_movies_pt(file_ids_ratings, file_ids_titles, min_rating, min_tokens=3):
    ids_ratings = {}
    ids_titles = {}
    with open(file_ids_ratings, 'r') as ratings:
        next(ratings)
        for line in ratings:
            cols = line.split('\t')
            ids_ratings[cols[0]] = float(cols[1])

    with open(file_ids_titles, 'r') as titles:
        next(titles)
        for line in titles:
            cols = line.split('\t')
            if cols[3] == 'PT' and len(cols[2].split()) >= min_tokens:
                ids_titles[cols[0]] = cols[2]

    titles = []
    for id in ids_ratings:
        if id in ids_titles and ids_ratings[id] >= min_rating:
            #print(ids_titles[id], ids_ratings[id])
            titles.append(ids_titles[id].lower())

    return titles


def check_movie_pt(movie_title, dict_forms_labels):
    movie_tokens = movie_title.lower().translate(str.maketrans('', '', string.punctuation)).split()
    verifier = 0
    if len(movie_tokens) <= 3 or movie_title[0:4] == 'Epis':
        return False
    for token in movie_tokens:
        if token in dict_forms_labels:
            label = dict_forms_labels[token]
            if check_pos(trim_pos(label[0][2])): #so' esta' a olhar para a primeira label
                verifier += 1
        else:
            return False
    if verifier < 2:
        return False
    for i in range(0, 10):
        if str(i) in movie_title:
            return False
    return True

def load_proverbs(file_proverbs):
    proverbs = []
    with open(file_proverbs, 'r') as file:
        for line in file:
            proverbs.append(line.strip())
    return proverbs


def create_expressions_file():
    movie_titles = best_rated_movies_pt('gen_inputs/title.ratings.tsv', 'gen_inputs/movie_titles_pt.tsv', 7, 3)
    print("Movie titles", len(movie_titles))

    dict_forms_labels = read_write_obj_file(1, None, '../models_db/dict_labels.pk1')

    filtered_titles = []
    for t in movie_titles:
        if check_movie_pt(t, dict_forms_labels):
            #        print(t)
            filtered_titles.append(t)
    print("Filtered titles", len(filtered_titles))

    proverbs = load_proverbs('gen_inputs/proverbs.txt')
    print("Proverbs", len(proverbs))

    expressions = []
    expressions.extend(proverbs)
    expressions.extend(filtered_titles)
    read_write_obj_file(0, expressions, '../models_db/prov_movies_v2.pk1')


if __name__ == '__main__':

    #create_expressions_file()

    '''
    expressions = read_write_obj_file(1, None, '../models_db/prov_movies_v2.pk1')
    for c, e in enumerate(expressions):
        if c < 1500:
            print(c, e.strip())
    '''

    '''
    print("Carregar labels...")
    all_labels = read_write_obj_file(1, None, '../models_db/list_labels_v3.pk1')
    print("Guardar dicionário labels...")
    labels_2_dict(all_labels)
    all_labels = read_write_obj_file(1, None, '../models_db/dict_labels.pk1')
    print(type(all_labels))

    print("Carregar GloVe...")
    model = KeyedVectors.load('../models_db/we_models/model_glove300.model')
    print("Guardar dicionário frequência...")
    create_freq_dict('gen_inputs/formas.cetempublico.utf8.txt', model, all_labels, min_freq=50)
    '''

    '''
    all_labels = read_write_obj_file(1, None, '../models_db/dict_labels.pk1')
    for k in all_labels:
        print(k, all_labels[k])'''

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
    #x = 'afeicão é manual'
    #print(x, x.lower().translate(str.maketrans('', '', string.punctuation)).strip())
