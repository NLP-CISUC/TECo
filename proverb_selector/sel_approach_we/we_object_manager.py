import time
from gensim.models import KeyedVectors
import NLPyPort as nlpyport
from proverb_selector.sel_utils.file_manager import *


def objects_creation():
    # news_data_retrieval()
    news = data_retrieval('../sel_inputs/newsTitles_naturee.txt', None)
    news_tokens = [nlpyport.tokenize_from_string(n) for n in news]
    news_tags = [nlpyport.tag(nt)[1] for nt in news_tokens]
    prov_tokens, prov_tags, prov_lemmas, prov_entidades = \
        nlpyport.full_pipe('../sel_inputs/proverbios_natura.txt')
    # prov_tokens = [tokenize_from_string(p) for p in proverbs]
    # prov_tags = [tag(p)[1] for p in prov_tokens]

    return news_tokens, news_tags, prov_tokens, prov_tags, prov_lemmas


def read_objects():
    news_tokens = read_write_obj_file(1, None, '../sel_outputs/news_tokens.pk1')
    news_tags = read_write_obj_file(1, None, '../sel_outputs/news_tags.pk1')
    prov_tokens = read_write_obj_file(1, None, '../sel_outputs/prov_tokens.pk1')
    prov_tags = read_write_obj_file(1, None, '../sel_outputs/prov_tags.pk1')
    return news_tokens, news_tags, prov_tokens, prov_tags


def write_objects(proverbs, news_tokens, news_tags, prov_tokens, prov_tags):
    read_write_obj_file(0, proverbs, '../sel_outputs/proverbs.pk1')
    read_write_obj_file(0, news_tokens, '../sel_outputs/news_tokens.pk1')
    read_write_obj_file(0, news_tags, '../sel_outputs/news_tags.pk1')
    read_write_obj_file(0, prov_tokens, '../sel_outputs/prov_tokens.pk1')
    read_write_obj_file(0, prov_tags, '../sel_outputs/prov_tags.pk1')
