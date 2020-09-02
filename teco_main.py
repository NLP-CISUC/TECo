import time

from gensim.models import KeyedVectors

from teco_config.load_config import load_config
from proverb_selector.sel_utils.file_manager import read_write_obj_file
from teco_twitterbot.twitter_bot import init_twitter_bot

if __name__ == '__main__':
    """
    all_headlines = read_write_obj_file(1, None, 'headline_gen/gen_inputs/sample_headlines.pk1')
    print(
        init_headline_generator_v2(
            headline=all_headlines[57],
            model=KeyedVectors.load('models_db/we_models/model_glove300.model'),
            all_headlines=read_write_obj_file(1, None, 'headline_gen/gen_inputs/sample_headlines.pk1'),
            all_proverbs=read_write_obj_file(1, None, 'models_db/prov_movies.pk1'),
            all_labels=read_write_obj_file(1, None, 'models_db/list_labels_v3.pk1'),
            gen_method=1, sel_method=2
        ))
    """
    # ----- CONFIGURATION -----
    # ----- expressions; output_db; we_model; lexicon; tfidf_words; tfidf_occur; 1st_sel_amount; sel_method; sleep_time
    file_paths = load_config()
    all_expressions = read_write_obj_file(1, None, file_paths[0])
    model = KeyedVectors.load(file_paths[1])
    all_labels = read_write_obj_file(1, None, file_paths[2])

    configs = [read_write_obj_file(1, None, file_paths[3]),
               read_write_obj_file(1, None, file_paths[4]),
               int(file_paths[5]),
               int(file_paths[6])]
    sleep_time = int(file_paths[7])
    method_order = file_paths[8]
    while True:
        start = time.time()
        init_twitter_bot(all_expressions=all_expressions, model=model, all_labels=all_labels, configs=configs,
                         method_order=method_order)
        time.sleep(sleep_time)
        end = time.time()
        # print("SLEEP TIME: ", end - start)
