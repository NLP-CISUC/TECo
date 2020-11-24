import time

from gensim.models import KeyedVectors

from teco_config.load_config import *
from proverb_selector.sel_utils.file_manager import read_write_obj_file
from teco_twitterbot.twitter_bot import init_twitter_bot
from teco_twitterbot.twitter_bot import call_teco

from gen_methods.selection_methods import init_prov_selector_standard

def test_twitter_bot(all_expressions, model, dict_forms_labels, dict_lemmas_labels, configs, gen_method):
    #while True:
    start = time.time()
    print("Init Twitter bot...")
    init_twitter_bot(all_expressions=all_expressions, model=model, dict_forms_labels=dict_forms_labels, dict_lemmas_labels=dict_lemmas_labels, configs=configs, gen_method=gen_method)
    print("Sleep...", sleep_time)
    time.sleep(sleep_time)
    end = time.time()
    # print("SLEEP TIME: ", end - start)

def test_headline_gen(all_expressions, model, dict_forms_labels, dict_lemmas_labels, configs, method_order):
    noticias_teste = [
        # um que não tem keywords
#        "BADABADA PA PA PIM",
        # um que so' da' com Subs
#        "Mourinho culpa-me por ter sido demitido do Chelsea. Esteve sempre contra mim",
#        "Alentejo: No Verão de 2021, o Redondo volta à alegria das Ruas Floridas",
        "Pareceu que o adversário queria passar mais do que nós à próxima eliminatória",
        "Covid-19: Proibição de circulação entre concelhos tem 10 exceções",
        #"MotoGP: Miguel Oliveira voa para a vitória no GP de Portugal",
        "Orçamento do Estado 2021: Finanças culpam PSD por coligações negativas que terão custado já mais de 20 milhões",
        #"Coronavírus: Estado de emergência: contactos limitados ao “mínimo indispensável” e celebrações até seis pessoas",
#        "Natal? É difícil antecipar. Privados prontos a libertar camas.",
#        "MotoGP: pilotos escolheram os melhores do ano e Miguel Oliveira destacou-se",
#        "Em 19 dias houve mais mortes por Covid do que nos últimos cinco meses"
    ]

    for n in noticias_teste:
        headline, generated = call_teco(n, all_expressions, model, dict_forms_labels, dict_lemmas_labels, configs, method_order)
        print(headline, "->", generated[1])

def test_console_gen(all_expressions, model, dict_forms_labels, dict_lemmas_labels, configs, method_order):
    while True:
        text = input("> ")
        headline, generated = call_teco(text, all_expressions, model, dict_forms_labels, dict_lemmas_labels, configs,
                                        method_order)
        print("> "+generated[1])


def dict_forms_to_lemmas_label(dict_forms_labels):
    dict_lemmas_labels = {}
    for form in dict_forms_labels:
        for entry in dict_forms_labels[form]:
            if entry[1] not in dict_lemmas_labels:
                dict_lemmas_labels[entry[1]] = []
            dict_lemmas_labels[entry[1]].append(entry)
    return dict_lemmas_labels

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
    print("Expressions", file_paths[EXPRESSIONS])
    all_expressions = read_write_obj_file(1, None, file_paths[EXPRESSIONS])
    print("Model", file_paths[EMBEDDINGS])
    model = KeyedVectors.load(file_paths[EMBEDDINGS])
    print("Lexicon", file_paths[LEXICON])
    dict_forms_labels = read_write_obj_file(1, None, file_paths[LEXICON])


    print("Frequencies", file_paths[FREQUENCIES])
    freqs = read_write_obj_file(1, None, file_paths[FREQUENCIES])

    #passar tf-idf para um dicionário
    #tfidf_words = read_write_obj_file(1, None, file_paths[TFIDF_WORDS])
    #tfidf_occurs = read_write_obj_file(1, None, file_paths[TFIDF_OCCURS])
    #tfidf = {}
    #for i in range(len(tfidf_words)):
    #    tfidf[tfidf_words[i]] = tfidf_occurs[i]

    #tfidf_words, tfidf_occur, first_sel_amount, sel_method
    configs = [freqs, int(file_paths[N_FIRST_SEL]), file_paths[FINAL_SEL]]

    gen_method = file_paths[GEN_METHOD]
    print("Gen method", gen_method)
    sleep_time = int(file_paths[TWEET_INTERVAL])
    print("Sleep time", sleep_time)

    dict_lemmas_labels = dict_forms_to_lemmas_label(dict_forms_labels)

    #test_console_gen(all_expressions, model, dict_forms_labels, dict_lemmas_labels, configs, gen_method)
    #test_headline_gen(all_expressions, model, dict_forms_labels, dict_lemmas_labels, configs, gen_method)
    test_twitter_bot(all_expressions, model, dict_forms_labels, dict_lemmas_labels, configs, gen_method)

    #selection = init_prov_selector_standard(0, ["Pareceu que o adversário queria passar mais do que nós à próxima eliminatória"], all_expressions, 10)
    #print(selection)