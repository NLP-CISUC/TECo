from pathlib import Path

start_path = Path(__file__).parent.absolute()

EXPRESSIONS="EXPRESSIONS"
EMBEDDINGS="EMBEDDINGS"
LEXICON="LEXICON"
FREQUENCIES="FREQUENCIES"
TFIDF_WORDS="TFIDF_WORDS"
TFIDF_OCCURS="TFIDF_OCCURS"
N_FIRST_SEL="N_FIRST_SEL"
FIRST_SEL="FIRST_SEL"
GEN_METHOD="GEN_METHOD"
FINAL_SEL="FINAL_SEL"
TWEET_INTERVAL="TWEET_INTERVAL"

def load_config(config_file='teco_config/config.properties'):
    config = {}
    with open(config_file, 'r') as configs:
        for row in configs:
            if row[0] != "#":
                cols = row.split("=")
                if cols[0] == EXPRESSIONS:
                    config[EXPRESSIONS] = cols[1].strip("\n")
                elif cols[0] == EMBEDDINGS:
                    config[EMBEDDINGS] = cols[1].strip("\n")
                elif cols[0] == LEXICON:
                    config[LEXICON] = cols[1].strip("\n")
                elif cols[0] == FREQUENCIES:
                    config[FREQUENCIES] = cols[1].strip("\n")
                elif cols[0] == TFIDF_WORDS:
                    config[TFIDF_WORDS] = cols[1].strip("\n")
                elif cols[0] == TFIDF_OCCURS:
                    config[TFIDF_OCCURS] = cols[1].strip("\n")
                elif cols[0] == N_FIRST_SEL:
                    config[N_FIRST_SEL] = cols[1].strip("\n")
                elif cols[0] == FIRST_SEL:
                    config[FIRST_SEL] = cols[1].strip("\n")
                elif cols[0] == FINAL_SEL:
                    config[FINAL_SEL] = cols[1].strip("\n")
                elif cols[0] == TWEET_INTERVAL:
                    config[TWEET_INTERVAL] = cols[1].strip("\n")
                elif cols[0] == GEN_METHOD:
                    config[GEN_METHOD] = cols[1].strip("\n").split(",")
        return config


def load_config_selector(config_file='teco_config/config.properties'):
    config_files = []
    with open(config_file, 'r') as configs:
        for row in configs:
            if row[0] != "#":
                if row.split("=")[0] == "Selector_cbow50":
                    config_files.append(row.split("=")[1].strip("\n"))
                elif row.split("=")[0] == "Selector_glove300":
                    config_files.append(row.split("=")[1].strip("\n"))
                elif row.split("=")[0] == "Selector_fasttext300":
                    config_files.append(row.split("=")[1].strip("\n"))
        return config_files
