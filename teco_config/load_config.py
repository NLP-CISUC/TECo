from pathlib import Path

start_path = Path(__file__).parent.absolute()


def load_config(config_file='teco_config/config.properties'):
    config_files = []
    with open(config_file, 'r') as configs:
        for row in configs:
            if row[0] != "#":
                if row.split("=")[0] == "Expressions_corpus_config_file":
                    config_files.append(row.split("=")[1].strip("\n"))
                elif row.split("=")[0] == "WE_model_config_file":
                    config_files.append(row.split("=")[1].strip("\n"))
                elif row.split("=")[0] == "Lexicon_config_file":
                    config_files.append(row.split("=")[1].strip("\n"))
                elif row.split("=")[0] == "Tfidf_words":
                    config_files.append(row.split("=")[1].strip("\n"))
                elif row.split("=")[0] == "Tfidf_occur":
                    config_files.append(row.split("=")[1].strip("\n"))
                elif row.split("=")[0] == "First_selection_amount":
                    config_files.append(row.split("=")[1].strip("\n"))
                elif row.split("=")[0] == "Sel_method":
                    config_files.append(row.split("=")[1].strip("\n"))
                elif row.split("=")[0] == "Tweet_interval":
                    config_files.append(row.split("=")[1].strip("\n"))
                elif row.split("=")[0] == "Method_order":
                    config_files.append(row.split("=")[1].strip("\n"))
        return config_files


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
