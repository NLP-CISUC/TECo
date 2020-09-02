import logging as log
import pickle
import csv
from newsapi.newsapi_client import NewsApiClient
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def data_retrieval(filename, encoding):
    with open(filename, 'r', encoding=encoding) as proverb_file:
        p_reader = proverb_file.readlines()
        # proverb_reader = csv.reader(proverb_file)
        proverbs = [row for row in p_reader]
        return proverbs


def read_write_obj_file(aux, obj, filename):
    if aux == 0:
        with open(filename, 'wb') as output:
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    else:
        with open(filename, 'rb') as i:
            my_obj = pickle.load(i)
        return my_obj

    return None


def get_tfidf_matrix(input_text, proverbs):
    vectorizer = TfidfVectorizer(use_idf=True)
    docs = input_text.copy()
    for p in proverbs:
        docs.append(p)
    return vectorizer, vectorizer.fit_transform(docs)


def get_word_tfidf(word, inp_id, vectorizer, tfidf_matrix):

    if word in vectorizer.get_feature_names():
        tmp = vectorizer.get_feature_names().index(word)
        print(tfidf_matrix.shape)
        # get the first vector out (for the first document)
        first_vector_tfidfvectorizer = tfidf_matrix[inp_id]
        # place tf-idf values in a pandas data frame
        df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=vectorizer.get_feature_names(),
                          columns=["tfidf"])
        df.sort_values(by=["tfidf"], ascending=False)

        return df.values.tolist()[tmp][0]
    else:
        return 0


def news_data_retrieval():
    newsapi = NewsApiClient(api_key='20e090e9c80047e2b990da48f070739b')
    csvFile = open('sel_inputs/ClimaGNews.csv', 'a')  # Open/Create a file to append data
    csvWriter = csv.writer(csvFile)  # Use csv Writer

    keywords = 'clima OR ambiente OR aquecimento global'
    # keywords = 'ronaldo'
    key_sources = 'www.dn.pt, publico.pt, expresso.pt, sapo.pt, www.jn.pt'
    # domínios inválidos : 'Noticiasaominuto.com, pt.euronews.com, natgeo.pt, www.sicnoticias.pt, ionline.sapo.pt'

    top_headlines = newsapi.get_everything(q=keywords,
                                           language='pt',
                                           page_size=100,
                                           sort_by='relevancy',
                                           domains=key_sources,
                                           from_param='2019-11-30')
    print(top_headlines)
    newstxt = ''
    for headline in top_headlines['articles']:
        csvWriter.writerow([headline['source']['name'], headline['title']])
        newstxt = newstxt + headline['title'] + '\n'
    # print(newstxt)
    file = open('sel_inputs/newsTitles_naturee.txt', 'w')
    file.write(newstxt)
    file.close()


def selector(input_text, proverbs, sim, info_sim, amount):
    # print("$$$$", input_text, proverbs, sim, info_sim)
    chosen_expressions = []
    for inp_id, inp in enumerate(input_text):
        for counter in range(amount):
            # print(counter, sim[0][inp_id])
            index_sim = sim[inp_id].index(max(sim[inp_id]))
            # print(inp, proverbs[index_sim], sim[inp_id][index_sim], info_sim)
            chosen_expressions.append((inp, proverbs[index_sim], sim[inp_id][index_sim]))
            if counter == 0:
                # LOGGING RESULTS
                log.info("[{}] Input: {}\n\tAt index: {} \n\tChosen proverb: {} \tSimilarity level: {}\n"
                         .format(info_sim, inp, index_sim, proverbs[index_sim], sim[inp_id][index_sim]))
            sim[inp_id].pop(index_sim)
    return chosen_expressions
