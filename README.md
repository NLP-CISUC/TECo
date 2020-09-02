# TECo_Project
 
# Download & Install
Files needed to put in folder 'models_db':
	- To put in folder 'bert_pretrained_models', download BERT MultiLingual Cased 'multi_cased_L-12_H-768_A-12' from: https://github.com/google-research/bert
	- To put in folder 'we_models', download from: 

# Configuration
Under 'teco_config', the configuration file is 'config.properties', where besides the paths to the models, there are also options for running TECo:
	- Adaptation method order: which runs first, VecDiff or Analogy
	- Amount of expressions to be selected from the corpus, before being adapted
	- Final Selection Method: 1 - TF-IDf; 2 - BERT;
	- Interval between tweets, in seconds. Useful if trying to run the twitter-bot.

# Running TECo
- Run 'bert_server_run.py'
- Run 'teco_main.py'