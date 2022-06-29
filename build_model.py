# -*- coding: utf-8 -*-
"""
Created on Wed Sep 1 2021
@author: Joe L/Joe HC/Steven H
This file contains the main code that is run to clean the text input and build the model. It then outputs that data
and model class into the pickle files for later use.


# get top 200 weighted terms for each topic
topic_terms_nmf = model_nmf.get_topic_terms(200,coincide = True)
# simplify to list of weight values for each term in each topic
topic_terms_nmf = [list(el.values()) for el in topic_terms_nmf]
fig = model_nmf.construct_topic_network(fnc.compare_topics(topic_terms_nmf),
                                        pet_topics_nmf)
fig.show()
############# Latent Semantic Analysis #############
lsa = TruncatedSVD(n_components=30,random_state=42)
model_lsa = fnc.TopicDecompositionSklearn(tfidf,lsa)
pet_topics = model_lsa.fit_transform(pet_text['title'])
############# Latent Dirichlet Allocation #############
lda = LatentDirichletAllocation(n_components=30,random_state=42)
model_lda = fnc.TopicDecompositionSklearn(tfidf,lda)
pet_topics_lda = model_lda.fit_transform_classify(pet_text['combined'],
                                                  max_topics = 2,
                                                  key_words = 10,
                                                  cut_off = 0.1)
# this gives a distance matrix where i,j is the cosine similarity of the mean embedding vectors for topics i and j
#emb = model.get_embedding_measure(30)
###########################################
##############################################
############# Data Visualization #############
vis_dir = r'.\figures\{}.html'
# plotly.express bar chart for number of documents per topic per priority
vis_data = pet_topics.groupby(['priority','topic'],as_index = False)['text'].agg({'count':'count'})
vis_data[['priority','topic']] = vis_data[['priority','topic']].astype(str)
fig = px.bar(vis_data,
             x = 'count',
             y = 'topic',
             color = 'priority',
             barmode = 'group',
             orientation = 'h',
             title = 'Number of documents mentioning each topic, by topic priority')
fig.write_html(r'.\figures\documents_per_topic_per_priority.html')
# plotly.express bar chart for weighting of top n words in each topic
for i in range(len(topics)):
    title = 'Topic ' + str(i) + ': Top ' + str(len(topics[i])) + ' LDA weighted key words'
    f = px.bar(x = topics[i].values(),
               y = topics[i].keys(),
               orientation = 'h',
               title = title)
    f.write_html(vis_dir.format(title.replace(':','').replace(' ','_')))
# plotly.graph_objects to build a network graph
fig = model.construct_topic_network(cosine_similarity, pet_topics, key_words_df=None)
fig.write_html(vis_dir.format(fig.layout.title.text.replace(' ','_')))
"""

###############################
########### Imports ###########

import pandas as pd
import spacy
from scripts.functions.utils import \
    ChainerCleanList, \
    multi_input

from scripts.functions.text_cleaning import \
    expand_contractions, \
    standardize_whitespace, \
    get_list_ngrams, \
    bigram_replacer, \
    translate_text

from scripts.functions.spacy_utils import SpacyTfidf

import pickle
import json
import sys
import logging
import argparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

from scripts.functions.model import compare_topics

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str)
parser.add_argument('--welsh_support', "-w", type=int)
parser.add_argument('--embedding', '-e', type=str)  # "glove" or "spacy"

# import plotly.express as px
# import plotly.graph_objs as go

# loaded to show plotly visuals in browser, necessary for spyder IDE
# import plotly.io as pio
# pio.renderers.default = 'browser'

###############################
path = "configs/base_config.json"
if path:
    try:
        config = json.load(open(path))
    except FileNotFoundError:
        print(
            '-- WARNING: User config file failed to load.\n')
    else:
        for key, value in config.items():
            config[key] = value
else:
    logging.warning("-- Custom user config were not found.")


def main():
    model = "nmf"
    welsh_support = 0

    args = parser.parse_args()

    if args.model:
        model = args.model
    else:
        model = "nmf"
        logging.info("defaulting to NMF - for LDA try `python build_model -m lda`")

    if args.welsh_support:
        welsh_support = args.welsh_support

    else:
        welsh_support = 0
        logging.info("welsh support off")

    if args.embedding:
        run_embedding = args.embedding
    else:
        run_embedding = None
        logging.info("Not generating embeddings for topics - use `-e spacy` or `-e glove`")

    ###
    logging.info("Loading data")
    ########### get data ###########
    df = pd.read_csv(r'data/petitions_sample_text.csv')

    df.fillna(
        {
            'action': '', 
            'background': '', 
            'additional_details': ''
        }, 
        inplace = True
    )

    df['combined'] = df['action'].str.cat(
        df[['background', 'additional_details']], 
        sep = ' '
    )

    if welsh_support == 1:
        logging.info('Translate to Welsh.')
        df['combined_welsh'] = df['combined'].apply(translate_text)
    else:
        logging.info('Welsh support not active.')
    
    ###
    logging.info('Loading text preprocessor using SpaCy language as a base.')
    ########### text preprocessing setup ###########
    preprocessor = ChainerCleanList(
        standardize_whitespace,
        expand_contractions
    )

    nlp = spacy.load('en_core_web_sm') # pre-trained pipeline
    
    tfidf = SpacyTfidf(
        nlp,
        preproc_function = preprocessor,
        stop_words = True,
        lemmatize = True,
        gensim_output = True
    )

    ###
    logging.info("Model Selection")
    ############## Model Selection ##############################

    if model == 'nmf':
        model = NMF(n_components=config['nmf']['n_components'],
                    beta_loss=config['nmf']['beta_loss'],
                    init=config['nmf']['init'],
                    max_iter=config['nmf']['max_iter'],
                    random_state=config['nmf']['random_state'])

    elif model == 'lda':
        model = LatentDirichletAllocation(random_state=config['lda']['random_state'])

    elif len(model) == 0:
        logging.warning("No model selected defaulting to NMF.")
        model = NMF(n_components=config['nmf']['n_components'],
                    beta_loss=config['nmf']['beta_loss'],
                    init=config['nmf']['init'],
                    max_iter=config['nmf']['max_iter'],
                    random_state=config['nmf']['random_state'])
    else:
        logging.error('Error. Please select a valid model: nmf or lda')
        return None

    updated_text = pet_text[['combined_bi', 'dummy_date']].copy()
    updated_text.rename(columns={
        'combined_bi': 'combined'
    }, inplace=True)

    model_set = TopicDecompositionSklearn(tfidf, model)
    pet_topics = model_set.fit_transform_classify(updated_text['combined'],
                                                  updated_text['dummy_date'],
                                                  max_topics=config['ftc']['max_topics'],
                                                  key_words=config['ftc']['key_words'],
                                                  topic_selector=config['ftc']['topic_selector'])
    logging.info("Model built")
    model_set.find_top_nouns(updated_text['combined'])
    logging.info("Found top nouns")
    model_set.fit_transform(updated_text['combined'])
    # model_lda.post_processing_mode()
    model_set.post_processing_mode()

    # Take the topic-term weights and run them through an embedding model

    if run_embedding:
        model_set.get_embedding_measure(10, mod=run_embedding)
        # print(compare_topics(model_set.topic_embedding_similarity))

    ######################### Load to pickles ##########################

    pickle_dir = r'data/pickles/{}.pkl'

    pet_topics.to_pickle(pickle_dir.format(r'petition_topics'), None)
    model_set.doc_topic_weights.to_pickle(pickle_dir.format(r'document_topics_weights'))

    with open(pickle_dir.format('model_set'), 'wb') as pkl:
        pickle.dump(model_set, pkl)
    pkl.close()
    logging.info('Pickles set up')


if __name__ == "__main__":
    main()
