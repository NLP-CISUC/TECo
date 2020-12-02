import time

from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer

from teco_config.load_config import *
from proverb_selector.sel_utils.file_manager import read_write_obj_file
from teco_twitterbot.twitter_bot import init_twitter_bot
from teco_twitterbot.twitter_bot import call_teco

from sklearn.metrics.pairwise import cosine_similarity

def test_twitter_bot(all_expressions, model, dict_forms_labels, dict_lemmas_labels, configs, gen_method, post=True):
    #start = time.time()
    print("Init Twitter bot...")
    while True:
        init_twitter_bot(all_expressions=all_expressions, model=model, dict_forms_labels=dict_forms_labels,
                         dict_lemmas_labels=dict_lemmas_labels, configs=configs, gen_method=gen_method, post=post)
        print("Sleep...", sleep_time)
        time.sleep(sleep_time)
    #end = time.time()
    # print("SLEEP TIME: ", end - start)

def test_headline_gen(all_expressions, model, dict_forms_labels, dict_lemmas_labels, configs, method_order):
    noticias_teste = [
        # um que não tem keywords
#        "BADABADA PA PA PIM",
        # um que so' da' com Subs
#        "Mourinho culpa-me por ter sido demitido do Chelsea. Esteve sempre contra mim",
#       "VÍDEO: erro de Otamendi e Rodrigo Pinho dá vantagem ao Marítimo",
        "Agência europeia confirma pedidos de vacinas e responde dentro de semanas",
        "Cristina Ferreira levou os amigos e tentou mudar tudo na TVI mas as derrotas sucessivas ameaçam o futuro",
        "Coronavírus: Profissionais de saúde sofreram estigma por parte de vizinhos",
        "Pareceu que o adversário queria passar mais do que nós à próxima eliminatória",
        "Covid-19: Proibição de circulação entre concelhos tem 10 exceções",
        "Liga dos Campeões: Noite de Champions com arte, fortuna e altruísmo de Bruno Fernandes",
#        "Alentejo: No Verão de 2021, o Redondo volta à alegria das Ruas Floridas",
        #"MotoGP: Miguel Oliveira voa para a vitória no GP de Portugal",
        "Orçamento do Estado 2021: Finanças culpam PSD por coligações negativas que terão custado já mais de 20 milhões",
        #"Coronavírus: Estado de emergência: contactos limitados ao “mínimo indispensável” e celebrações até seis pessoas",
#        "Natal? É difícil antecipar. Privados prontos a libertar camas.",
#        "MotoGP: pilotos escolheram os melhores do ano e Miguel Oliveira destacou-se",
#        "Em 19 dias houve mais mortes por Covid do que nos últimos cinco meses"
    ]

    for n in noticias_teste:
        headline, generated = call_teco(n, all_expressions, model, dict_forms_labels, dict_lemmas_labels, configs, method_order)
        if generated:
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
    configs = [freqs, file_paths[FIRST_SEL], int(file_paths[N_FIRST_SEL]), file_paths[FINAL_SEL]]

    gen_method = file_paths[GEN_METHOD]
    print("Gen method", gen_method)
    sleep_time = int(file_paths[TWEET_INTERVAL])
    print("Sleep time", sleep_time)

    dict_lemmas_labels = dict_forms_to_lemmas_label(dict_forms_labels)

    #test_console_gen(all_expressions, model, dict_forms_labels, dict_lemmas_labels, configs, gen_method)
    test_headline_gen(all_expressions, model, dict_forms_labels, dict_lemmas_labels, configs, gen_method)
    #test_twitter_bot(all_expressions, model, dict_forms_labels, dict_lemmas_labels, configs, gen_method, post=True)

    # vectorizer = TfidfVectorizer(max_df=0.75, min_df=2)
    # vectors = vectorizer.fit_transform(all_expressions)
    # print(vectors.shape)
    # features = vectorizer.get_feature_names()
    # #print(len(features))
    #
    # new_exp = ["eu não gosto de brincar contigo", "as armas e os barões assinalados", "no caminho do bem",
    #            "esta vida são dois dias e um é para acordar", "as vidas não têm prazo de validade", "piloto obrigado a mudar de número",
    #            "tudo na vida tem uma consequência", "como um vício que liberta todo o bom e mau em mim"]
    # vectors_new = vectorizer.transform(new_exp)
    # features = vectorizer.get_feature_names()
    # print(vectors_new.shape)
    #
    # a_vectors = vectors.toarray()
    # a_vectors_new = vectors_new.toarray()
    #
    # for c1, v in enumerate(a_vectors_new):
    #     #print(v)
    #     max_id = -1
    #     max_sim = -1
    #     for c2, vn in enumerate(a_vectors):
    #         sim = cosine_similarity([v], [vn])[0][0]
    #         if sim > max_sim:
    #             max_sim = sim
    #             max_id = c2
    #     print(sim, "*", new_exp[c1], "*", all_expressions[max_id])

    #selection = init_prov_selector_standard(0, ["Pareceu que o adversário queria passar mais do que nós à próxima eliminatória"], all_expressions, 10)
    #print(selection)