import time
import random
import tweepy
from gensim.models import KeyedVectors

from headline_gen.headline_gen import init_headline_generator_v2
from proverb_selector.sel_utils.file_manager import read_write_obj_file
from teco_twitterbot.twitter_utils.twitter_manager import *


def init_twitter():
    CONSUMER_KEY = 'b3RgdN5VSO6RvdmJY9aaW07zl'
    CONSUMER_SECRET = 'wagRXKmvsp9A40LZ2htl0E8amVUgYbi9wIMTDoyhdbU271OPTs'
    ACCESS_KEY = '1246561294808989700-tKTdL79oq4pnVGjQWP2pYkoyoLIegY'
    ACCESS_SECRET = 'WWdRPH3aeehYK3inbemulQoki5jwEE9hXjxMqaRVsR6I6'
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_KEY, ACCESS_SECRET)
    return tweepy.API(auth)


def init_twitter_bot(all_expressions, model, all_labels, configs, method_order):
    api = init_twitter()
    og_tweets = tweet_retrieval(api=api, amount=10)
    all_headlines = [twt[2] for twt in og_tweets]

    for counter, og_tweet in enumerate(og_tweets):
        # gen_method = random.randint(1, 2)
        # first_sel = random.randint(1, 2)
        final_sel = random.randint(1, 2)

        if not check_headline(og_tweet[2], all_labels):
            continue

        headline, gen_expression = call_teco(headline=og_tweet[2], all_headlines=all_headlines,
                                             all_expressions=all_expressions, model=model, all_labels=all_labels,
                                             configs=configs, method_order=method_order)
        if headline is None or gen_expression is None:
            continue
        # Upper case letter
        gen_expression = gen_expression.lower().capitalize()
        new_tweet = gen_expression + '\n' + 'https://twitter.com/' + str(og_tweet[0]) + '/status/' + str(og_tweet[1])
        new_tweet_len = len(new_tweet)

        if 0 < new_tweet_len < 280:
            # print("NEW TWEET ", new_tweet, new_tweet_len)
            try:
                api.update_status(status=new_tweet)
                return
            except tweepy.TweepError:
                print(tweepy.TweepError)


def call_teco(headline, all_headlines, all_expressions, model, all_labels, configs, method_order):
    chosen_headline = ""
    expression = ""
    if method_order == 0:
        chosen_headline, expression, similarity = init_headline_generator_v2(
            headline=headline, all_headlines=all_headlines, all_expressions=all_expressions,
            model=model, all_labels=all_labels, configs=configs, gen_method=2
        )
        # if vecdif does not produce results, use Analogy
        if expression == "-":
            chosen_headline, expression, similarity = init_headline_generator_v2(
                headline=headline, all_headlines=all_headlines, all_expressions=all_expressions,
                model=model, all_labels=all_labels, configs=configs, gen_method=1
            )
        # if neither vecdif nor analogy produce results, use Substitution
        if expression == "-":
            chosen_headline, expression, similarity = init_headline_generator_v2(
                headline=headline, all_headlines=all_headlines, all_expressions=all_expressions,
                model=model, all_labels=all_labels, configs=configs, gen_method=0
            )
    elif method_order == 1:
        chosen_headline, expression, similarity = init_headline_generator_v2(
            headline=headline, all_headlines=all_headlines, all_expressions=all_expressions,
            model=model, all_labels=all_labels, configs=configs, gen_method=1
        )
        # if analogy does not produce results, use vecdiff
        if expression == "-":
            chosen_headline, expression, similarity = init_headline_generator_v2(
                headline=headline, all_headlines=all_headlines, all_expressions=all_expressions,
                model=model, all_labels=all_labels, configs=configs, gen_method=2
            )
        # if neither vecdif nor analogy produce results, use Substitution
        if expression == "-":
            chosen_headline, expression, similarity = init_headline_generator_v2(
                headline=headline, all_headlines=all_headlines, all_expressions=all_expressions,
                model=model, all_labels=all_labels, configs=configs, gen_method=0
            )

    if chosen_headline != "" and expression != "":
        return headline, expression
    return None, None
