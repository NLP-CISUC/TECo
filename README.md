# TECo - Texto Em Contexto (Text in Context)
This system aims to, given a short-text input, adapt and select Portuguese expressions (e.g., proverbs and movie titles) in order to approximate them to the input, enhancing relatedness, originality and, possibly, funniness.

# Download & Install
Files needed to put in folder 'models_db':
- To put in folder 'bert_pretrained_models', download <a href="https://github.com/google-research/bert">BERT MultiLingual Cased 'multi_cased_L-12_H-768_A-12'</a>
- To put in folder 'we_models', downloadable <a href="https://drive.google.com/drive/folders/1oCVCjoAED2DErrVCuk3yi0MFVvX3BO02?usp=sharing">here</a>

# Configuration
Under 'teco_config', the configuration file is 'config.properties', where besides the paths to the models, there are also options for running TECo:
- Adaptation methods and their order, split by commas: VecDiff, Analogy, Subs
- Amount of expressions to be selected from the corpus, for adaptation
- Final Selection Method: TF-IDF, BERT
- Interval between tweets, in seconds. Useful if trying to run the twitter-bot.

# Running TECo
- Run 'bert_server_run.py' (only if BERT is required)
- Run 'teco_main.py'

# Additional information
- TECo is described in two research papers:

<i>TECo: Exploring Word Embeddings for Text Adaptation to a given Context</i>, included in the proceedings of the  <a href="http://computationalcreativity.net/iccc20/papers/ICCC20_Proceedings.pdf">11th International Conference on Computational Creativity</a>, which can be cited as follows:
<pre>
@inproceedings{mendes_goncalooliveira:iccc2020b,
	author = {Rui Mendes and Hugo {Gon{\c c}alo Oliveira}},
	booktitle = {Proceedings of the 11th International Conference on Computational Creativity, September 7-11, 2020, Coimbra},
	pages = {185--188},
	publisher = {ACC},
	series = {ICCC 2020},
	title = {TECo: Exploring Word Embeddings for Text Adaptation to a given Context},
	year = {2020}}
</pre>

<i>Amplifying the Range of News Stories with Creativity: Methods and their Evaluation, in Portuguese</i>, included in the proceedings of the <a href="https://www.aclweb.org/anthology/2020.inlg-1.32/">13th International Conference on Natural Language Generation</a>, which can be cited as follows:
<pre>
@inproceedings{mendes-goncalo-oliveira-2020-amplifying,
    title = "Amplifying the Range of News Stories with Creativity: Methods and their Evaluation, in {P}ortuguese",
    author = "Mendes, Rui and Gon{\c{c}}alo Oliveira, Hugo",
    booktitle = "Proceedings of the 13th International Conference on Natural Language Generation",
    month = dec,
    year = "2020",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.inlg-1.32",
    pages = "252--262",
}
</pre>
