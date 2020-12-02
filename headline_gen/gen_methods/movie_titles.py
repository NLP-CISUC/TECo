import csv
import string

from proverb_selector.sel_utils.file_manager import *
from headline_gen.gen_utils.utils_gen import trim_pos, check_pos


def check_label(token, all_labels):
    if token in all_labels:
        return all_labels[token]
    return []


def check_movie_pt(movie_title, all_labels):
    movie_tokens = movie_title.lower().translate(str.maketrans('', '', string.punctuation)).split()
    verifier = 0
    if len(movie_tokens) <= 3 or movie_title[0:4] == 'Epis':
        return False
    for token in movie_tokens:
        label = check_label(token, all_labels)
        if not label:
            return False
        elif check_pos(trim_pos(label[0][2])): #so' esta' a olhar para a primeira label
            verifier += 1
    if verifier < 2:
        return False
    for i in range(0, 10):
        # print(i)
        if str(i) in movie_title:
            return False
    return True


def init_movie_retrieval():
    portuguese_movies = []
    all_labels = read_write_obj_file(1, None, 'models_db/list_labels_v3.pk1')
    with open('models_db/movietitles.txt', 'rb') as movie_file:
        movie_reader = movie_file.readlines()
        for counter, row in enumerate(movie_reader):
            if counter < round(len(movie_reader)/4):
                continue
            line = row.decode().split('\t')
            if 'PT' in line[3] and check_movie_pt(line[2], all_labels):
                print(row)
                # print(line[2])
                portuguese_movies.append(line[2]+'\n')
    portuguese_movies.sort()
    with open('headline_gen/gen_inputs/movietitles_filtered_v3.txt', 'w') as movie_file:
        for title in portuguese_movies:
            movie_file.write(title)
    print(len(portuguese_movies))
    read_write_obj_file(0, portuguese_movies, 'headline_gen/gen_inputs/movietitles_v3.pk1')


