import time
import random
import tweepy
from gensim.models import KeyedVectors

from headline_gen.headline_gen import *
from proverb_selector.sel_utils.file_manager import read_write_obj_file
from proverb_selector.sel_approach_standard.standard_approach import init_prov_selector_we
from teco_twitterbot.twitter_utils.twitter_manager import *
from headline_gen.gen_methods.selection_methods import *

def auth_twitter(twitter_conf):

    auth = tweepy.OAuthHandler(twitter_conf[1], twitter_conf[2])
    auth.set_access_token(twitter_conf[3], twitter_conf[4])
    return tweepy.API(auth)


def load_twitter_confs(twitter_conf):
    with open(twitter_conf, 'r') as tconf:
        for row in tconf:
            if row[0] != "#":
                cols = row.split("=")
                if cols[0] == "TWEET_INTERVAL":
                    sleep = int(cols[1].strip("\n"))
                elif cols[0] == "CONSUMER_KEY":
                    consumer_key = cols[1].strip("\n")
                elif cols[0] == "CONSUMER_SECRET":
                    consumer_secret = cols[1].strip("\n")
                elif cols[0] == "ACCESS_KEY":
                    access_key = cols[1].strip("\n")
                elif cols[0] == "ACCESS_SECRET":
                    access_secret = cols[1].strip("\n")

    return sleep, consumer_key, consumer_secret, access_key, access_secret


def run_twitter_bot(all_expressions, model, dict_forms_labels, dict_lemmas_labels, configs, gen_method, twitter_conf, post=True):

    api = auth_twitter(twitter_conf)

    og_tweets = tweet_retrieval(api=api, amount=10)
    #all_headlines = [twt[2] for twt in og_tweets]

    print("Tweets", len(og_tweets))

    for counter, og_tweet in enumerate(og_tweets):
        # gen_method = random.randint(1, 2)
        # first_sel = random.randint(1, 2)

        if not check_headline(og_tweet[2], dict_forms_labels):
            continue

        headline, gen_expression = call_teco(headline=og_tweet[2].strip(),
                                             all_expressions=all_expressions, model=model,
                                             dict_forms_labels=dict_forms_labels, dict_lemmas_labels=dict_lemmas_labels,
                                             configs=configs, gen_method=gen_method)

        print("Gen expression", gen_expression)
        if not headline or not gen_expression:
            continue

        if post:
            # Upper case letter
            post_text = gen_expression[1].lower().capitalize()
            new_tweet = post_text + '\n' + 'https://twitter.com/' + str(og_tweet[0]) + '/status/' + str(og_tweet[1])
            new_tweet_len = len(new_tweet)

            if 0 < new_tweet_len < 280:
                # print("NEW TWEET ", new_tweet, new_tweet_len)
                try:
                    api.update_status(status=new_tweet)
                    return
                except tweepy.TweepError:
                    print(tweepy.TweepError)


def call_teco(headline, all_expressions, model, dict_forms_labels, dict_lemmas_labels, configs, gen_method):
    # Structure of Headline: [([0] word; [1] lemma; [2] PoS tag; [3] Form), .. , ()]
    headline_tokens = get_tokens(headline)
    #print("Headline tokens: ", headline_tokens)

    first_sel_method, first_sel_amount, sel_method = configs

    # -------- First Selection -----------
    selected_proverbs = first_selection(first_sel_method, first_sel_amount//2, first_sel_amount//2, headline, all_expressions, model)
    print("Adapting ", len(selected_proverbs), "proverbs")
    longer_expressions, shorter_expressions = split_longer_shorter_expressions(selected_proverbs)

    # -------- Generation -----------
    nlpyport.load_config() #required for PoS tagging
    headline_keywords = get_headline_keywords(headline, headline_tokens, dict_forms_labels, model, min=1, max=5)
    all_generated = None
    for method in gen_method:
        print("GENERATING with "+method+" ...")

        #for Analogy and VecDiff use only longer expressions
        expressions = selected_proverbs if method == SUBSTITUTION else longer_expressions
        all_generated = headline_generator_v2(
            headline=headline, use_expressions=expressions,
            model=model, dict_forms_labels=dict_forms_labels, dict_lemmas_labels=dict_lemmas_labels, gen_method=method,
            headline_keywords=headline_keywords, shorter_expressions=shorter_expressions
        )
        if all_generated:
            break;

    if not all_generated:
        print("[INFO] Could not generate any expression.")
        return None, None

    #print("GENERATED=", all_generated)
    # -------- Final Selection -----------
    if sel_method == TFIDF or sel_method == WE or sel_method == BERT:
        ranked = final_rank(headline, all_generated, sel_method, model=model, all_expressions=all_expressions, headline_tokens=headline_tokens)
        #print("RANKED=", ranked)
        if ranked:
            print("* ", len(ranked), "GENERATED with", method, ":")
            for i in range(6):
                if i < len(ranked):
                    print(i, ranked[i])
            return headline, ranked[0]
        else:
            print("[ERROR] Invalid selection method.")
            return None, None


def first_selection(sel_method, quant_tfidf, quant_random, headline, all_expressions, model):
    first_sel = get_first_selection(all_expressions, headline, sel_method, amount=quant_tfidf, model=model)
    selected_expressions = [exp[1] for exp in first_sel]
    # BERT selection
    '''
    selected_proverbs.extend(
        [prov[1] for prov in get_first_selection(all_expressions, headline, 2,
                                                 amount=amount, model=model)])
    '''
    #print("similar:", len(selected_expressions), selected_expressions)
    random_extension = random.sample(all_expressions, k=quant_random)
    selected_expressions.extend(random_extension)
    #print("random:", len(random_extension), random_extension)
    #print("all:", len(selected_expressions))
    '''
    extension_counter = 0
    for counter in range(len(random_extension)):
        if random_extension[counter] not in selected_proverbs:
            selected_proverbs.append(random_extension[counter])
            extension_counter += 1
        if extension_counter >= amount:
            break
    '''
    return list(set(selected_expressions))


def split_longer_shorter_expressions(all_expressions, short_length=5):
    longer_expressions = []
    shorter_expressions = []
    for exp in all_expressions:
        exp_tokens = get_tokens(exp)
        if len(exp_tokens) > short_length:
            longer_expressions.append(exp)
        else:
            shorter_expressions.append(exp)
    return longer_expressions, shorter_expressions


def final_rank(headline, gen_expressions, method=TFIDF, n=10, all_expressions=None, model=None, headline_tokens=None):

    if method == BERT:
        selected_proverbs = init_prov_selector_bert_service(headline, gen_expressions, amount=n)
    elif method == WE:
        selected_proverbs = init_prov_selector_we(headline, gen_expressions, model=model, input_tokens=headline_tokens, tfidf=True, corpus=all_expressions, amount=n)
    else:
        selected_proverbs = init_prov_selector_standard(0, headline, gen_expressions, amount=n, corpus=all_expressions)

    if not selected_proverbs:
        return None
    #print("SELECTED PROVERBS "+method+":\t", len(selected_proverbs), selected_proverbs[0], '\n')
    return selected_proverbs