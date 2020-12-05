#!/usr/bin/python
# -*- coding: utf-8 -*-

import csv
import time
import numpy as np
import sklearn.metrics as met
import krippendorff
from gensim.models import KeyedVectors
from scipy import stats
from sel_approach_standard.standard_approach import init_prov_selector_standard
from sel_approach_transformer.transformer_approach import init_prov_selector_bert
from headline_gen.headline_gen import headline_generator_v2
from sel_utils.file_manager import data_retrieval
from NLPyPort.FullPipeline import load_config
from gen_utils.utils_gen import *
from proverb_selector.sel_utils.file_manager import *


def storage():
    """
        init_headline_generator_v2(hl, model, all_headlines, prov_movies, all_labels,
                                   gen_method=0, sel_methods=[1, 1]),
        init_headline_generator_v2(hl, model, all_headlines, prov_movies, all_labels,
                                   gen_method=0, sel_methods=[1, 2]),
        init_headline_generator_v2(hl, model, all_headlines, prov_movies, all_labels,
                                   gen_method=0, sel_methods=[2, 1]),
        init_headline_generator_v2(hl, model, all_headlines, prov_movies, all_labels,
                                   gen_method=0, sel_methods=[2, 2]),
        init_headline_generator_v2(hl, model, all_headlines, prov_movies, all_labels,
                                   gen_method=1, sel_methods=[1, 1]),
        init_headline_generator_v2(hl, model, all_headlines, prov_movies, all_labels,
                                   gen_method=1, sel_methods=[1, 2]),
        init_headline_generator_v2(hl, model, all_headlines, prov_movies, all_labels,
                                   gen_method=1, sel_methods=[2, 1]),
        init_headline_generator_v2(hl, model, all_headlines, prov_movies, all_labels,
                                   gen_method=1, sel_methods=[2, 2]),
        init_headline_generator_v2(hl, model, all_headlines, prov_movies, all_labels,
                                   gen_method=2, sel_methods=[1, 1]),
        init_headline_generator_v2(hl, model, all_headlines, prov_movies, all_labels,
                                   gen_method=2, sel_methods=[1, 2]),
        init_headline_generator_v2(hl, model, all_headlines, prov_movies, all_labels,
                                   gen_method=2, sel_methods=[2, 1]),
        init_headline_generator_v2(hl, model, all_headlines, prov_movies, all_labels,
                                   gen_method=2, sel_methods=[2, 2]),
        init_prov_selector_standard(0, [hl], prov_movies, amount=1)[0][1],
        init_prov_selector_bert(7, [hl], prov_movies, amount=1)[0][1]"""


def decode_index(index):
    if index == 0:
        return ['Substitution', 'TFIDFVectorizer']
    elif index == 1:
        return ['Substitution', 'BERT']
    elif index == 2:
        return ['Analogy', 'TFIDFVectorizer']
    elif index == 3:
        return ['Analogy', 'BERT']
    elif index == 4:
        return ['Comp_Vec', 'TFIDFVectorizer']
    elif index == 5:
        return ['Comp_Vec', 'BERT']
    elif index == 6:
        return ['-', 'TFIDFVectorizer']
    elif index == 7:
        return ['-', 'BERT']


def sample_data_creation():
    load_config()
    all_headlines_iter = read_write_obj_file(
        1, None, 'gen_inputs/sample_headlines.pk1')
    prov_movies = read_write_obj_file(1, None, 'gen_inputs/prov_movies.pk1')
    all_labels = read_write_obj_file(1, None, 'gen_inputs/list_labels_v3.pk1')
    model = KeyedVectors.load('C:/Disciplinas/Tese/ThesisWork/proverb_selector/sel_inputs/we_models/model_glove300.model')
    all_data = []
    for counter in range(len(all_headlines_iter)):
        hl = all_headlines_iter[counter]
        all_headlines = all_headlines_iter.copy()
        print("#####\t[HEADLINE]\t", counter, hl, all_headlines[counter])
        print("#####\t [ALL HEADLINES]\t", all_headlines, all_headlines_iter)
        new_data = [
            headline_generator_v2(hl, model, all_headlines, prov_movies, all_labels,
                                  gen_method=0, sel_method=1),
            headline_generator_v2(hl, model, all_headlines, prov_movies, all_labels,
                                  gen_method=0, sel_method=2),
            headline_generator_v2(hl, model, all_headlines, prov_movies, all_labels,
                                  gen_method=1, sel_method=1),
            headline_generator_v2(hl, model, all_headlines, prov_movies, all_labels,
                                  gen_method=1, sel_method=2),
            headline_generator_v2(hl, model, all_headlines, prov_movies, all_labels,
                                  gen_method=2, sel_method=1),
            headline_generator_v2(hl, model, all_headlines, prov_movies, all_labels,
                                  gen_method=2, sel_method=2),
            init_prov_selector_standard(0, [hl], prov_movies, amount=1)[0][1],
            init_prov_selector_bert(7, [hl], prov_movies, amount=1)[0][1]
        ]
        all_data.append(new_data)
        read_write_obj_file(0, all_data, 'gen_outputs/sample_data.pk1')
        del new_data
        # gc.collect()
        time.sleep(30)


def init_sample_tester():
    sample_data_creation()
    all_prov_movies = read_write_obj_file(1, None, 'gen_inputs/prov_movies.pk1')
    data = read_write_obj_file(1, None, 'gen_outputs/sample_data.pk1')
    print(all_prov_movies)

    with open('gen_outputs/thesis_data.csv', 'w') as csv_file:
        field_names = ['Headline', 'Generated_Expression', 'Generation_Method', 'Final_Selection']
        csv_writer = csv.DictWriter(csv_file, fieldnames=field_names, dialect='excel')
        csv_writer.writeheader()

        for index_headline, headline_all in enumerate(data):
            index_counter = 0
            for headline_det in headline_all:
                print(headline_det, type(headline_det))
                if len(headline_det) == 1:
                    headline_det = headline_det[0]
                if type(headline_det) == tuple:
                    headline, gen_exp, sim = headline_det
                elif type(headline_det) == str:
                    gen_exp = headline_det
                info = decode_index(index_counter)
                csv_writer.writerow({'Headline': headline, 'Generated_Expression': gen_exp,
                                     'Generation_Method': info[0],
                                     'Final_Selection': info[1]})
                index_counter += 1


def create_base_dict(dict_or_array):
    combinations = [('Substitution', 'TFIDFVectorizer', 'TFIDFVectorizer'), ('Substitution', 'TFIDFVectorizer', 'BERT'),
                    ('Substitution', 'BERT', 'TFIDFVectorizer'), ('Substitution', 'BERT', 'BERT'),
                    ('Analogy', 'TFIDFVectorizer', 'TFIDFVectorizer'), ('Analogy', 'TFIDFVectorizer', 'BERT'),
                    ('Analogy', 'BERT', 'TFIDFVectorizer'), ('Analogy', 'BERT', 'BERT'),
                    ('Comp_Vec', 'TFIDFVectorizer', 'TFIDFVectorizer'), ('Comp_Vec', 'TFIDFVectorizer', 'BERT'),
                    ('Comp_Vec', 'BERT', 'TFIDFVectorizer'), ('Comp_Vec', 'BERT', 'BERT'),
                    ('-', '-', 'TFIDFVectorizer'),
                    ('-', '-', 'BERT')
                    ]
    if not dict_or_array:
        return combinations
    dict_comb = {}
    for comb in combinations:
        dict_comb[comb] = []

    return dict_comb


def decode_relatedness(value):
    if value == 1:
        return 1
    elif value == 2:
        return 1
    elif value == 3:
        return 2
    elif value == 4:
        return 3
    elif value == 5:
        return 3
    else:
        return 1


def decode_comb(comb, method_sel):
    if method_sel == 0:
        if comb[0] == 'Substitution':
            return 0
        elif comb[0] == 'Analogy':
            return 1
        elif comb[0] == 'Comp_Vec':
            return 2
    elif method_sel == 1:
        if comb[2] == 'TFIDFVectorizer':
            return 1
        elif comb[2] == 'BERT':
            return 0
    return -1


def init_sample_interpreter():
    all_data = create_base_dict(dict_or_array=True)
    with open('gen_inputs/final_sample_results.csv', 'r', errors='ignore') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            line = dict(row)
            if line['Headline'] != '' and line['Generated_Expression'] != '-':
                all_data[(line['Generation_Method'], line['First_Selection'], line['Final_Selection'])].append(
                    [line['Hugo_Syntax'], line['Hugo_Relatedness'], line['Hugo_Fun'],
                     line['Rui_Syntax'], line['Rui_Relatedness'], line['Rui_Fun']]
                )
    all_combinations = create_base_dict(dict_or_array=False)
    print(all_combinations)
    stats_combinations = create_base_dict(dict_or_array=True)
    with open('gen_outputs/final_sample_stats.csv', 'w', errors='ignore') as csv_file:
        field_names = ['Combination', 'Avg_Syntax', 'Std_Syntax', 'Median_Syntax', 'Mode_Syntax',
                       'Avg_Relatedness', 'Std_Relatedness', 'Median_Relatedness', 'Mode_Relatedness',
                       'Avg_Fun', 'Std_Fun', 'Median_Fun', 'Mode_Fun',
                       'Kappa_Syntax', 'Kappa_Relatedness', 'Kappa_Fun',
                       'Krippendorff_Syntax', 'Krippendorff_Relatedness', 'Krippendorff_Fun']
        csv_writer = csv.DictWriter(csv_file, fieldnames=field_names, dialect='excel')
        csv_writer.writeheader()
        total_syntax = []
        total_hugo_syntax = []
        total_rui_syntax = []
        total_relatedness = []
        total_hugo_relatedness = []
        total_rui_relatedness = []
        total_fun = []
        total_hugo_fun = []
        total_rui_fun = []

        total_subs_syntax = []
        total_subs_relatedness = []
        total_subs_fun = []
        total_analogy_syntax = []
        total_analogy_relatedness = []
        total_analogy_fun = []
        total_compvec_syntax = []
        total_compvec_relatedness = []
        total_compvec_fun = []
        bert_syntax = []
        bert_rel = []
        bert_fun = []
        tfidf_syntax = []
        tfidf_rel = []
        tfidf_fun = []

        for comb in all_combinations:
            comb_data = all_data[comb]
            method_aux = decode_comb(comb, 0)
            sel_aux = decode_comb(comb, 1)
            print(comb, method_aux, sel_aux)
            comb_syntax = []
            comb_relatedness = []
            comb_fun = []
            hugo_syntax = []
            rui_syntax = []
            hugo_relatedness = []
            rui_relatedness = []
            hugo_fun = []
            rui_fun = []

            for tmp in comb_data:
                if tmp[2] == '-' or tmp[2] == '':
                    continue
                tmp_hugo = int(tmp[0])
                tmp_rui = int(tmp[3])
                total_syntax.extend([tmp_hugo, tmp_rui])
                comb_syntax.extend([tmp_hugo, tmp_rui])
                hugo_syntax.append(tmp_hugo)
                total_hugo_syntax.append(tmp_hugo)
                rui_syntax.append(tmp_rui)
                total_rui_syntax.append(tmp_rui)
                if method_aux == 0:
                    total_subs_syntax.extend([tmp_hugo, tmp_rui])
                elif method_aux == 1:
                    total_analogy_syntax.extend([tmp_hugo, tmp_rui])
                elif method_aux == 2:
                    total_compvec_syntax.extend([tmp_hugo, tmp_rui])
                if sel_aux == 0:
                    bert_syntax.extend([tmp_hugo, tmp_rui])
                elif sel_aux == 1:
                    tfidf_syntax.extend([tmp_hugo, tmp_rui])

                tmp_hugo = decode_relatedness(int(tmp[1]))
                tmp_rui = decode_relatedness(int(tmp[4]))
                total_relatedness.extend([tmp_hugo, tmp_rui])
                comb_relatedness.extend([tmp_hugo, tmp_rui])
                hugo_relatedness.append(tmp_hugo)
                total_hugo_relatedness.append(tmp_hugo)
                rui_relatedness.append(tmp_rui)
                total_rui_relatedness.append(tmp_rui)
                if method_aux == 0:
                    total_subs_relatedness.extend([tmp_hugo, tmp_rui])
                elif method_aux == 1:
                    total_analogy_relatedness.extend([tmp_hugo, tmp_rui])
                elif method_aux == 2:
                    total_compvec_relatedness.extend([tmp_hugo, tmp_rui])
                if sel_aux == 0:
                    bert_rel.extend([tmp_hugo, tmp_rui])
                elif sel_aux == 1:
                    tfidf_rel.extend([tmp_hugo, tmp_rui])

                tmp_hugo = int(tmp[2])
                tmp_rui = int(tmp[5])
                total_fun.extend([tmp_hugo, tmp_rui])
                comb_fun.extend([tmp_hugo, tmp_rui])
                hugo_fun.append(tmp_hugo)
                total_hugo_fun.append(tmp_hugo)
                rui_fun.append(tmp_rui)
                total_rui_fun.append(tmp_rui)
                if method_aux == 0:
                    total_subs_fun.extend([tmp_hugo, tmp_rui])
                elif method_aux == 1:
                    total_analogy_fun.extend([tmp_hugo, tmp_rui])
                elif method_aux == 2:
                    total_compvec_fun.extend([tmp_hugo, tmp_rui])
                if sel_aux == 0:
                    bert_fun.extend([tmp_hugo, tmp_rui])
                elif sel_aux == 1:
                    tfidf_fun.extend([tmp_hugo, tmp_rui])

            stats_combinations[comb] = [np.average(comb_syntax), np.std(comb_syntax),
                                        np.median(comb_syntax), stats.mode(comb_syntax)[0][0],
                                        np.average(comb_relatedness), np.std(comb_relatedness),
                                        np.median(comb_relatedness), stats.mode(comb_relatedness)[0][0],
                                        np.average(comb_fun), np.std(comb_fun),
                                        np.median(comb_fun), stats.mode(comb_fun)[0][0],
                                        met.cohen_kappa_score(hugo_syntax, rui_syntax),
                                        met.cohen_kappa_score(hugo_relatedness, rui_relatedness),
                                        met.cohen_kappa_score(hugo_fun, rui_fun),
                                        krippendorff.alpha(value_counts=np.array([hugo_syntax, rui_syntax]),
                                                           level_of_measurement='nominal'),
                                        krippendorff.alpha(value_counts=np.array([hugo_relatedness, rui_relatedness]),
                                                           level_of_measurement='nominal'),
                                        krippendorff.alpha(value_counts=np.array([hugo_fun, rui_fun]),
                                                           level_of_measurement='nominal'),
                                        ]

            csv_writer.writerow({'Combination': comb, 'Avg_Syntax': stats_combinations[comb][0],
                                 'Std_Syntax': stats_combinations[comb][1],
                                 'Median_Syntax': stats_combinations[comb][2],
                                 'Mode_Syntax': stats_combinations[comb][3],
                                 'Avg_Relatedness': stats_combinations[comb][4],
                                 'Std_Relatedness': stats_combinations[comb][5],
                                 'Median_Relatedness': stats_combinations[comb][6],
                                 'Mode_Relatedness': stats_combinations[comb][7],
                                 'Avg_Fun': stats_combinations[comb][8],
                                 'Std_Fun': stats_combinations[comb][9],
                                 'Median_Fun': stats_combinations[comb][10],
                                 'Mode_Fun': stats_combinations[comb][11],
                                 'Kappa_Syntax': stats_combinations[comb][12],
                                 'Kappa_Relatedness': stats_combinations[comb][13],
                                 'Kappa_Fun': stats_combinations[comb][14],
                                 'Krippendorff_Syntax': stats_combinations[comb][15],
                                 'Krippendorff_Relatedness': stats_combinations[comb][16],
                                 'Krippendorff_Fun': stats_combinations[comb][17],
                                 })

        csv_writer.writerow({'Combination': 'Total_Substitution',
                             'Avg_Syntax': np.average(total_subs_syntax),
                             'Std_Syntax': np.std(total_subs_syntax),
                             'Median_Syntax': np.median(total_subs_syntax),
                             'Mode_Syntax': stats.mode(total_subs_syntax)[0][0],
                             'Avg_Relatedness': np.average(total_subs_relatedness),
                             'Std_Relatedness': np.std(total_subs_relatedness),
                             'Median_Relatedness': np.median(total_subs_relatedness),
                             'Mode_Relatedness': stats.mode(total_subs_relatedness)[0][0],
                             'Avg_Fun': np.average(total_subs_fun),
                             'Std_Fun': np.std(total_subs_fun),
                             'Median_Fun': np.median(total_subs_fun),
                             'Mode_Fun': stats.mode(total_subs_fun)[0][0],
                             'Kappa_Syntax': '-', 'Kappa_Relatedness': '-', 'Kappa_Fun': '-',
                             'Krippendorff_Syntax': '-', 'Krippendorff_Relatedness': '-', 'Krippendorff_Fun': '-',
                             })

        csv_writer.writerow({'Combination': 'Total_Analogy',
                             'Avg_Syntax': np.average(total_analogy_syntax),
                             'Std_Syntax': np.std(total_analogy_syntax),
                             'Median_Syntax': np.median(total_analogy_syntax),
                             'Mode_Syntax': stats.mode(total_analogy_syntax)[0][0],
                             'Avg_Relatedness': np.average(total_analogy_relatedness),
                             'Std_Relatedness': np.std(total_analogy_relatedness),
                             'Median_Relatedness': np.median(total_analogy_relatedness),
                             'Mode_Relatedness': stats.mode(total_analogy_relatedness)[0][0],
                             'Avg_Fun': np.average(total_analogy_fun),
                             'Std_Fun': np.std(total_analogy_fun),
                             'Median_Fun': np.median(total_analogy_fun),
                             'Mode_Fun': stats.mode(total_analogy_fun)[0][0],
                             'Kappa_Syntax': '-', 'Kappa_Relatedness': '-', 'Kappa_Fun': '-',
                             'Krippendorff_Syntax': '-', 'Krippendorff_Relatedness': '-', 'Krippendorff_Fun': '-',
                             })

        csv_writer.writerow({'Combination': 'Total_CompVec',
                             'Avg_Syntax': np.average(total_compvec_syntax),
                             'Std_Syntax': np.std(total_compvec_syntax),
                             'Median_Syntax': np.median(total_compvec_syntax),
                             'Mode_Syntax': stats.mode(total_compvec_syntax)[0][0],
                             'Avg_Relatedness': np.average(total_compvec_relatedness),
                             'Std_Relatedness': np.std(total_compvec_relatedness),
                             'Median_Relatedness': np.median(total_compvec_relatedness),
                             'Mode_Relatedness': stats.mode(total_compvec_relatedness)[0][0],
                             'Avg_Fun': np.average(total_compvec_fun),
                             'Std_Fun': np.std(total_compvec_fun),
                             'Median_Fun': np.median(total_compvec_fun),
                             'Mode_Fun': stats.mode(total_compvec_fun)[0][0],
                             'Kappa_Syntax': '-', 'Kappa_Relatedness': '-', 'Kappa_Fun': '-',
                             'Krippendorff_Syntax': '-', 'Krippendorff_Relatedness': '-', 'Krippendorff_Fun': '-',
                             })


        csv_writer.writerow({'Combination': 'Total_BERT',
                             'Avg_Syntax': np.average(bert_syntax), 'Std_Syntax': np.std(bert_syntax),
                             'Median_Syntax': np.median(bert_syntax), 'Mode_Syntax': stats.mode(bert_syntax)[0][0],
                             'Avg_Relatedness': np.average(bert_rel), 'Std_Relatedness': np.std(bert_rel),
                             'Median_Relatedness': np.median(bert_rel), 'Mode_Relatedness': stats.mode(bert_rel)[0][0],
                             'Avg_Fun': np.average(bert_fun), 'Std_Fun': np.std(bert_fun),
                             'Median_Fun': np.median(bert_fun), 'Mode_Fun': stats.mode(bert_fun)[0][0],
                             'Kappa_Syntax': '-', 'Kappa_Relatedness': '-', 'Kappa_Fun': '-',
                             'Krippendorff_Syntax': '-', 'Krippendorff_Relatedness': '-', 'Krippendorff_Fun': '-',
                             })

        csv_writer.writerow({'Combination': 'Total_TFIDF',
                             'Avg_Syntax': np.average(tfidf_syntax), 'Std_Syntax': np.std(tfidf_syntax),
                             'Median_Syntax': np.median(tfidf_syntax), 'Mode_Syntax': stats.mode(tfidf_syntax)[0][0],
                             'Avg_Relatedness': np.average(tfidf_rel), 'Std_Relatedness': np.std(tfidf_rel),
                             'Median_Relatedness': np.median(tfidf_rel),
                             'Mode_Relatedness': stats.mode(tfidf_rel)[0][0],
                             'Avg_Fun': np.average(tfidf_fun), 'Std_Fun': np.std(tfidf_fun),
                             'Median_Fun': np.median(tfidf_fun), 'Mode_Fun': stats.mode(tfidf_fun)[0][0],
                             'Kappa_Syntax': '-', 'Kappa_Relatedness': '-', 'Kappa_Fun': '-',
                             'Krippendorff_Syntax': '-', 'Krippendorff_Relatedness': '-', 'Krippendorff_Fun': '-',
                             })

        csv_writer.writerow({'Combination': 'Total',
                             'Avg_Syntax': np.average(total_syntax),
                             'Std_Syntax': np.std(total_syntax),
                             'Median_Syntax': np.median(total_syntax),
                             'Mode_Syntax': stats.mode(total_syntax)[0][0],
                             'Avg_Relatedness': np.average(total_relatedness),
                             'Std_Relatedness': np.std(total_relatedness),
                             'Median_Relatedness': np.median(total_relatedness),
                             'Mode_Relatedness': stats.mode(total_relatedness)[0][0],
                             'Avg_Fun': np.average(total_fun),
                             'Std_Fun': np.std(total_fun),
                             'Median_Fun': np.median(total_fun),
                             'Mode_Fun': stats.mode(total_fun)[0][0],
                             'Kappa_Syntax': met.cohen_kappa_score(total_hugo_syntax, total_rui_syntax),
                             'Kappa_Relatedness': met.cohen_kappa_score(total_hugo_relatedness, total_rui_relatedness),
                             'Kappa_Fun': met.cohen_kappa_score(total_hugo_fun, total_rui_fun),
                             'Krippendorff_Syntax': krippendorff.alpha(
                                 value_counts=np.array([total_hugo_syntax, total_rui_syntax]),
                                 level_of_measurement='nominal'),
                             'Krippendorff_Relatedness': krippendorff.alpha(
                                 value_counts=np.array([total_hugo_relatedness, total_rui_relatedness]),
                                 level_of_measurement='nominal'),
                             'Krippendorff_Fun': krippendorff.alpha(
                                 value_counts=np.array([total_hugo_fun, total_rui_fun]),
                                 level_of_measurement='nominal'),
                             })

        print("---------------------------------")
        print("Total_Sub_Syntax: ", count_proportions(total_subs_syntax, len(total_subs_syntax)))
        print("Total_Analogy_Syntax: ", count_proportions(total_analogy_syntax, len(total_analogy_syntax)))
        print("Total_Sub_Syntax: ", count_proportions(total_compvec_syntax, len(total_compvec_syntax)))
        print("Total_Syntax: ", count_proportions(total_syntax, len(total_syntax)))
        print("tfidf_syntax: ", count_proportions(tfidf_syntax, len(tfidf_syntax)), len(tfidf_syntax))
        print("bert_syntax: ", count_proportions(bert_syntax, len(bert_syntax)))
        print("---------------------------------")
        print("total_subs_relatedness: ", count_proportions(total_subs_relatedness, len(total_subs_relatedness)))
        print("total_analogy_relatedness: ", count_proportions(total_analogy_relatedness, len(total_analogy_relatedness)))
        print("total_compvec_relatedness: ", count_proportions(total_compvec_relatedness, len(total_compvec_relatedness)))
        print("total_relatedness: ", count_proportions(total_relatedness, len(total_relatedness)))
        print("tfidf_rel: ", count_proportions(tfidf_rel, len(tfidf_rel)))
        print("bert_rel: ", count_proportions(bert_rel, len(bert_rel)))
        print("---------------------------------")
        print("total_subs_fun: ", count_proportions(total_subs_fun, len(total_subs_fun)))
        print("total_analogy_fun: ", count_proportions(total_analogy_fun, len(total_analogy_fun)))
        print("total_compvec_fun: ", count_proportions(total_compvec_fun, len(total_compvec_fun)))
        print("total_fun: ", count_proportions(total_fun, len(total_fun)))
        print("tfidf_fun: ", count_proportions(tfidf_fun, len(tfidf_fun)))
        print("bert_fun: ", count_proportions(bert_fun, len(bert_fun)))

        # print(len(total_hugo_syntax), len(total_hugo_relatedness), len(total_hugo_fun))
        # print(len(total_rui_syntax), len(total_rui_relatedness), len(total_rui_fun))


def count_proportions(to_be_counted, max_size):
    return to_be_counted.count(1)/max_size * 100, \
           to_be_counted.count(2)/max_size * 100, \
           to_be_counted.count(3)/max_size * 100


def init_prepare_forms():
    data = read_write_obj_file(1, None, 'gen_outputs/all_thesis_data.pk1')

    with open('gen_outputs/forms_data.csv', 'w') as csv_file:
        field_names = ['Question']
        csv_writer = csv.DictWriter(csv_file, fieldnames=field_names, dialect='excel')
        csv_writer.writeheader()

        for index_headline, headline_all in enumerate(data):
            index_counter = 0
            for headline_det in headline_all:
                if len(headline_det) == 1:
                    headline_det = headline_det[0]
                if type(headline_det) == tuple:
                    headline, gen_exp, sim = headline_det
                elif type(headline_det) == str:
                    gen_exp = headline_det
                info = decode_index(index_counter)
                if index_counter == 0:
                    question = 'Considerando o título de notícia: "' + headline.strip() + '"'
                    csv_writer.writerow({'Question': question})
                question = 'Classifique a expressão: "' + gen_exp.strip() + '"'
                csv_writer.writerow({'Question': question})
                index_counter += 1


if __name__ == '__main__':
    load_config()
    start = time.time()
    # sample_headlines = data_retrieval('gen_inputs/sample_headlines.txt', None)
    # gen_read_write_obj_file(0, sample_headlines, 'gen_inputs/sample_headlines.pk1')
    # all_prov_movies = data_retrieval('gen_inputs/prov_and_movies.txt', None)
    # read_write_obj_file(0, all_prov_movies, 'gen_inputs/prov_movies.pk1')
    # init_sample_tester()
    init_sample_interpreter()
    init_prepare_forms()

    end = time.time()
    print("TOTAL EXECUTION TIME: ", end - start)
