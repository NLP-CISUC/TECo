import time
import random
import tweepy
from gensim.models import KeyedVectors

from headline_gen.headline_gen import init_headline_generator_v2
from proverb_selector.sel_utils.file_manager import read_write_obj_file
from teco_twitterbot.twitter_utils.twitter_manager import *
from gen_methods.selection_methods import *

def init_twitter():
    CONSUMER_KEY = 'b3RgdN5VSO6RvdmJY9aaW07zl'
    CONSUMER_SECRET = 'wagRXKmvsp9A40LZ2htl0E8amVUgYbi9wIMTDoyhdbU271OPTs'
    ACCESS_KEY = '1246561294808989700-ADXyzmCMsFacrqb2vy9eh0yfTHSHot'
    ACCESS_SECRET = 'nORnUxVSmq1rXwlguqrDPRW7k1vIew8YxokxOFrqO3gBk'
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_KEY, ACCESS_SECRET)
    return tweepy.API(auth)


def init_twitter_bot(all_expressions, model, dict_forms_labels, dict_lemmas_labels, configs, gen_method):

    api = init_twitter()

    og_tweets = tweet_retrieval(api=api, amount=10)
    #all_headlines = [twt[2] for twt in og_tweets]

    print("Tweets", len(og_tweets))

    for counter, og_tweet in enumerate(og_tweets):
        # gen_method = random.randint(1, 2)
        # first_sel = random.randint(1, 2)

        if not check_headline(og_tweet[2], dict_forms_labels):
            continue

        headline, gen_expression = call_teco(headline=og_tweet[2],
                                             all_expressions=all_expressions, model=model,
                                             dict_forms_labels=dict_forms_labels, dict_lemmas_labels=dict_lemmas_labels,
                                             configs=configs, gen_method=gen_method)

        print("Gen expression", gen_expression)
        if not headline or not gen_expression:
            continue

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
    headline_tokens = headline.lower().translate(str.maketrans('', '', string.punctuation)).split()

    tfidf, first_sel_amount, sel_method = configs

    # -------- First Selection -----------
    selected_proverbs = first_selection(first_sel_amount//2, first_sel_amount//2, headline, all_expressions, model)
    print("Adapting ", len(selected_proverbs), "proverbs")

    # -------- Generation -----------
    all_generated = None
    for method in gen_method:
        print("GENERATING with "+method+" ...")
        all_generated = init_headline_generator_v2(
            headline=headline, headline_tokens=headline_tokens, use_expressions=selected_proverbs,
            model=model, dict_forms_labels=dict_forms_labels, dict_lemmas_labels=dict_lemmas_labels, tfidf=tfidf, gen_method=method
        )
        if all_generated:
            break;

    if not all_generated:
        print("[INFO] Could not generate any expression.")
        return None, None

    #print("GENERATED=", all_generated)
    # -------- Final Selection -----------
    if sel_method == TFIDF or sel_method == BERT:
        ranked = final_rank(headline, all_generated, sel_method, all_expressions=all_expressions)
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


def first_selection(quant_tfidf, quant_random, headline, all_expressions, model):
    selected_proverbs = [prov[1] for prov in get_first_selection(all_expressions, headline, TFIDF, amount=quant_tfidf, model=model)]
    # BERT selection
    '''
    selected_proverbs.extend(
        [prov[1] for prov in get_first_selection(all_expressions, headline, 2,
                                                 amount=amount, model=model)])
    '''
    #print(len(selected_proverbs))
    random_extension = random.sample(all_expressions, k=quant_random)
    selected_proverbs.extend(random_extension)
    #print(len(selected_proverbs))
    '''
    extension_counter = 0
    for counter in range(len(random_extension)):
        if random_extension[counter] not in selected_proverbs:
            selected_proverbs.append(random_extension[counter])
            extension_counter += 1
        if extension_counter >= amount:
            break
    '''
    return list(set(selected_proverbs))


def final_rank(headline, gen_expressions, method=TFIDF, n=10, all_expressions=None):
    #selected_proverbs = []
    if method == BERT:
        selected_proverbs = init_prov_selector_bert(7, [headline], gen_expressions, amount=n)
    else:
        selected_proverbs = init_prov_selector_standard(0, [headline], gen_expressions, amount=n, corpus=all_expressions)

    if not selected_proverbs:
        return None
    #print("SELECTED PROVERBS "+method+":\t", len(selected_proverbs), selected_proverbs[0], '\n')
    return selected_proverbs