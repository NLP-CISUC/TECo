import csv
import string

import numpy as np
import sklearn.metrics as met
import statsmodels.stats.inter_rater as fk
import krippendorff
from scipy import stats

from proverb_selector.sel_utils.file_manager import read_write_obj_file


def read_sel_stats():
    rel_total = []
    fun_total = []
    kappa_rel_total = []
    kappa_fun_total = []
    answer_total = 0
    filename = 'sel_outputs/selector_results/all_form_responses.csv'
    with open(filename, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row_id, data in enumerate(csv_reader):
            # print(len(data), data)
            if row_id in [0, 5, 10, 15, 20, 25]:
                if row_id != 0:
                    total_fleiss_rel = []
                    total_fleiss_fun = []
                    for c in range(min(len(tmp_kappa_rel[0]), len(tmp_kappa_fun[1]))):
                        fleiss_rel = [0, 0, 0, 0]
                        fleiss_fun = [0, 0, 0, 0]

                        for row_rel in tmp_kappa_rel:
                            if row_rel[c] == 1:
                                fleiss_rel[0] += 1
                            elif row_rel[c] == 2:
                                fleiss_rel[1] += 1
                            elif row_rel[c] == 3:
                                fleiss_rel[2] += 1
                            elif row_rel[c] == 4:
                                fleiss_rel[3] += 1
                        total_fleiss_rel.append(fleiss_rel)

                        for row_fun in tmp_kappa_fun:
                            if row_fun[c] == 1:
                                fleiss_fun[0] += 1
                            elif row_fun[c] == 2:
                                fleiss_fun[1] += 1
                            elif row_fun[c] == 3:
                                fleiss_fun[2] += 1
                            elif row_fun[c] == 4:
                                fleiss_fun[3] += 1
                        total_fleiss_fun.append(fleiss_fun)

                    print(total_fleiss_rel)
                    kappa_rel_total.append(fk.fleiss_kappa(np.array(total_fleiss_rel), method='fleiss'))
                    kappa_fun_total.append(fk.fleiss_kappa(np.array(total_fleiss_fun), method='fleiss'))
                    # tmp = np.array([[1, 2], [2, 1], [1, 2]])
                    # print("###################################", krippendorff.alpha(np.array(tmp_kappa_rel)))
                questions = data
                answer_row_rel = []
                tmp_kappa_rel = []
                answer_row_fun = []
                tmp_kappa_fun = []
            else:
                answer_total += 1
                for question_id, answer in enumerate(data):
                    print(questions[question_id], question_id, answer)
                    if questions[question_id] in ['1', '2', '3', '4', '5', '6']:
                        continue
                    if 'avaliaria' in questions[question_id] and 'entre' in questions[question_id]:
                        print("." + answer + ".")
                        rel_total.append(int(answer))
                        answer_row_rel.append(int(answer))
                    elif 'cada' in questions[question_id] and 'Relacionando' in questions[question_id]:
                        fun_total.append(int(answer))
                        answer_row_fun.append(int(answer))
                    else:
                        continue
                if answer_row_rel:
                    tmp_kappa_rel.append(answer_row_rel)
                if answer_row_fun:
                    tmp_kappa_fun.append(answer_row_fun)
                answer_row_rel = []
                answer_row_fun = []

    print(len(rel_total), len(fun_total))
    print(np.average(kappa_rel_total), np.average(kappa_fun_total))
    print(answer_total)
    return rel_total, fun_total


def selector_shared_tokens():
    dict_sel_methods = {'Jaccard': [], 'CountVectorizer': [], 'TFIDFVectorizer': [], 'WE+Glove': [], 'WE+Glove+TFIDF': [],
                        'WE+FT': [], 'WE+FT+TFIDF': [], 'BERT': []}
    with open('sel_outputs/selector_results/sel_results.csv', 'r',
              errors='ignore', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file, dialect='excel')
        counter = 0
        for row in csv_reader:
            counter += 1
            headline_toks = row['Headline'].lower().translate(str.maketrans('', '', string.punctuation)).split()
            for key in dict_sel_methods.keys():
                tmp_counter = 0
                key_toks = row[key].lower().translate(str.maketrans('', '', string.punctuation)).split()
                print(key, headline_toks, key_toks)
                for tok in key_toks:
                    if tok in headline_toks:
                        tmp_counter += 1
                dict_sel_methods[key].append(tmp_counter)

    print(dict_sel_methods)
    for key in dict_sel_methods.keys():
        print(key, np.average(dict_sel_methods[key]), np.std(dict_sel_methods[key]))


if __name__ == '__main__':
    # rel, fun = read_sel_stats()
    selector_shared_tokens()
