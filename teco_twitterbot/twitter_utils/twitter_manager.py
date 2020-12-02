import csv
import string

from gen_methods.movie_titles import check_label
from gen_utils.utils_gen import get_tokens
from proverb_selector.sel_utils.file_manager import read_write_obj_file
import tweepy


def trim_tweet(tweet_text):
    tmp_twt = tweet_text.split('http')[0]
    if 'RT' in tmp_twt.split():
        tmp_twt = tmp_twt.split(":")[1:]
        if len(tmp_twt) == 1:
            tmp_twt = tmp_twt[0].strip()
        else:  # in case there is more than one ':' in the sentence
            for t in tmp_twt:
                tmp_twt = tmp_twt + ':' + t

    if '\n' not in tmp_twt:
        tmp_twt = tmp_twt + '\n'
    return tmp_twt


def tweet_retrieval(api, amount):
    # multiply amount to ensure that you have the wanted amount, in case some tweets fail conditions
    public_tweets = api.home_timeline(count=amount*2)
    my_user = api.me()
    current_tweets = []
    for tweet in public_tweets:
        dict_tweet = tweet.__dict__
        if dict_tweet['lang'] == 'pt' and dict_tweet['author'].id != my_user.id:
            tmp_tweet = trim_tweet(tweet.text)
            # ----- Structure -> (author_id, tweet_id, tweet_text) -----
            current_tweets.append((dict_tweet['author'].id, dict_tweet['id_str'], tmp_tweet))
            if len(current_tweets) == amount:
                break

    # read_write_obj_file(0, current_tweets, 'teco_twitterbot/twitter_input/current_tweets_og.pk1')
    return current_tweets


def get_info_methods(l_methods):
    used_methods = []
    # Generation Methods
    if l_methods[0] == 0:
        used_methods.append('Substitution')
    elif l_methods[0] == 1:
        used_methods.append('Analogy')
    elif l_methods[0] == 2:
        used_methods.append('VecDiff')

    # Final Selection Method
    if l_methods[1] == 0:
        used_methods.append('WE')
    elif l_methods[1] == 1:
        used_methods.append('TFIDFVectorizer')
    elif l_methods[1] == 2:
        used_methods.append('BERT')

    return used_methods


def find_og_tweet(tweet, og_twts):
    for twt in og_twts:
        if tweet[:-2] in twt[2]:  # not considering the final \n
            return twt
    return None


def get_selection_results(filename):
    with open(filename, newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        selection_data = list(csv_reader)[1:]
    return selection_data


def check_headline(headline, all_labels):
    headline_tokens = get_tokens(headline)
    total_tok = len(headline_tokens)
    if total_tok < 5:
        return False
    error_tok = 0
    for token in headline_tokens:
        label = check_label(token, all_labels)
        if not label:
            error_tok += 1

    if error_tok/total_tok > 0.3:
        return False
    return True
