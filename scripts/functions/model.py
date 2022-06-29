# -*- coding: utf-8 -*-
"""
Created on Wed Sep 1 2021

@author: Joe L/Joe HC/Steven H

The topic decomposition class is built here with all the functions for model building and outputting the topic
classified df.

"""

import pandas as pd
import numpy as np
from math import pi ,sin ,cos
import plotly.express as px
import plotly.graph_objs as go
import spacy
from collections import Counter
import logging
import json

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV

from gensim.matutils import Sparse2Corpus
from gensim.models.nmf import Nmf, CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from operator import itemgetter

from scripts.functions.utils import PlotlyColorMap


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


class TopicDecompositionSklearn:

    def __init__(self, tfidf=None, decomposer=None):

        if len(set([tfidf, decomposer])) == 2 and None in set([tfidf, decomposer]):
            raise ValueError('tfidf and decomposer must be either both None or neither None')

        self.tfidf = tfidf
        self.decomposer = decomposer

        # Check whether Topic Decomposition is to be used for visuals and
        # post processing or for analyzing new text
        self.turn_method_off = False
        if tfidf is None and decomposer is None:
            self.turn_method_off = True

        self.topic_term_weights = None
        self.doc_topic_weights = None
        self.docs_unclassified = None
        self.top_nouns = None
        self.top_scored_words = None

    def all_topic_term_weights(self):

        """
            Returns list of 'term : weight' dictionaries for each topic. list
            index corresponds to topic index. Term-weight dictionaries ordered
            by descending weights
        """

        if self.turn_method_off:
            return None

        output = []

        terms = self.get_feature_names()

        for index, component in enumerate(self.decomposer.components_):
            zipped = sorted(zip(terms, component),
                            key=lambda x: x[-1],
                            reverse=True)
            to_append = {k: v for k, v in zipped}

            output.append(to_append)

        return output

    def fit_transform(self, texts, store=False, topic_selector=30):

        if self.turn_method_off:
            return None

        vectors = self.tfidf.fit_transform(texts)

        df_tfidfvect = pd.DataFrame(data = vectors.toarray(), columns = self.tfidf.get_feature_names())
        word_scores = pd.DataFrame(df_tfidfvect.sum(axis=0), columns=['score'])

        # returning the top scored words from tfidf
        word_rank = word_scores.sort_values(by=['score'])
        word_rank['word'] = word_rank.index
        word_rank = word_rank.reset_index()
        limit = 0-config['top_words']['words']
        self.top_scored_words = [word for word in word_rank['word'][limit:]]

        # check if use if NMF or LDA
        if isinstance(topic_selector, int):
            self.decomposer.n_components = topic_selector

        elif topic_selector == "coherence":
            # process text for gensim models and then apply their n_components in sklearn
            corpus = Sparse2Corpus(vectors, documents_columns = False)
            topic_nums = list(np.arange(10, 100 + 1, 5))
            coherence_scores = []

            if isinstance(self.decomposer, NMF):
                for num in topic_nums:
                    nmf = Nmf(
                        corpus = corpus,
                        num_topics=num,
                        id2word= {v: k for k, v in self.tfidf.vocabulary_.items()},
                        normalize=True,
                        random_state=config['gensim']['nmf']['random_state']
                    )

                    cm = CoherenceModel(
                        model=nmf,
                        corpus = corpus,
                        dictionary = {v: k for k, v in self.tfidf.vocabulary_.items()},
                        coherence='c_v'
                    )

                    coherence_scores.append(round(cm.get_coherence(), 5))
                    print("Topics tested: " + str(num) + ". Coherence: " + str(round(cm.get_coherence(), 5)))

                # score and choose best can be farmed out to a function
                scores = list(zip(topic_nums, coherence_scores))
                best_num_topics = sorted(scores, key=itemgetter(1), reverse=True)[0][0]
                print("Best num_topics, NMF: " + str(best_num_topics))
                self.decomposer.n_components = best_num_topics

            elif isinstance(self.decomposer, LatentDirichletAllocation):
                for num in topic_nums:
                    lda = LdaModel(
                        corpus=corpus,
                        num_topics=num,
                        id2word= {v: k for k, v in self.tfidf.vocabulary_.items()},
                        random_state=config['gensim']['lda']['random_state']
                    )

                    cm = CoherenceModel(
                        model=lda,
                        corpus=corpus,
                        dictionary = {v: k for k, v in self.tfidf.vocabulary_.items()},
                        coherence='c_v'
                    )

                    coherence_scores.append(round(cm.get_coherence(), 5))

                # score and choose best can be farmed out to a function
                scores = list(zip(topic_nums, coherence_scores))
                best_num_topics = sorted(scores, key=itemgetter(1), reverse=True)[0][0]
                print("Best num_topics, LDA: " + str(best_num_topics))
                self.decomposer.n_components = best_num_topics

        else:
            return None

        output = self.decomposer.fit_transform(vectors)

        # check whether output should be stored in self.doc_topic_weights
        if store:
            self.doc_topic_weights = pd.DataFrame(output)

        return output

    def fit_transform_classify(self, texts, dates, max_topics=1,
                               key_words=0, cut_off=None, topic_selector=30):

        """ Determines threshold for a document-topic weight to be considered
            significant. Threshold is determined by max_topics and cut-off.

            Returns DataFrame of document-topic pairs. DataFrame contains
            document text; topic index; topic priority - a rank value of the
            documents' topic weight; document-topic weighting; and (optionally)
            the top 'key_words' number of term-topic weighted terms.

            texts: list-like entity of text data to fit_transform.

            max_topics: int or 'dynamic'. The maximum number of topics to
            assign to each documnet. Order of assignment follows document-topic
            weights.

            key_words: int. Return the top term-topic weighted terms for each
            topic assigned to a text.

            cut_off: float. Specifies the document-topic weighting threshold
            required to be returned. If None then the top 'max_topics' number
            of document-topic pairs will be returned for each document.
        """

        if self.turn_method_off:
            return None

        # Checking Arguments passed to Parameters
        if type(max_topics) is str and max_topics != 'dynamic':
            raise ValueError('max_topics must be int or "dynamic"')

        if type(max_topics) is int:

            if max_topics <= 0:
                raise ValueError('max_topics must be at least 1')

            if max_topics > self.decomposer.n_components:
                raise ValueError('max_topics must be less than or equal to number of topics.')

        if key_words < 0:
            raise ValueError('key_words must be at least 0')

        # Call fit_transform function above
        self.doc_topic_weights = pd.DataFrame(self.fit_transform(texts, topic_selector=topic_selector))

        # output contains texts argument in a dataframe
        output = pd.concat([texts, dates], axis=1)
        output.rename(columns={output.columns[0]: 'text', output.columns[1]:'dates'}, inplace=True)

        # number of rows and columns in topic_weights, used in
        # upcomming 'for' loops
        rows, columns = self.doc_topic_weights.shape

        # The following code will determine the significant topic
        # weight value required for each topic to classify a text
        # as a specifc topic
        if max_topics == 'dynamic':

            # Initialize instance of Decicision tree regressor
            splitter = DecisionTreeRegressor(max_depth=1)

            # flatten doc_topic_weights to list
            all_weights = pd.DataFrame(self.doc_topic_weights.stack().values)

            # Train splitter and retrieve cut point from base node's threshold
            splitter.fit(all_weights, all_weights[0])

            cut_off = splitter.tree_.threshold[0]

            # the following dictionary and list will be used to create a dataframe
            interim = {'topic': [], 'weight': [], 'priority': []}
            row_idx = []

            # iterate through each documents' topic weights
            for i in range(rows):
                doc_weights = self.doc_topic_weights.loc[i]
                temp = []

                # iterate through each topic weight
                for j in range(columns):
                    # create a list of topic, document-topic weight pairs
                    temp.append((j, doc_weights[j]))

                # sort topics and select by weights in decending order
                temp = sorted(temp, key=lambda el: el[1], reverse=True)
                for j in range(len(temp)):
                    interim['topic'].append(temp[j][0])
                    interim['weight'].append(temp[j][1])
                    interim['priority'].append(j + 1)  # Rank of topic relevance
                    row_idx.append(i)  # Equal to input text's index in output df

            output = pd.merge(output, pd.DataFrame(interim, row_idx),
                              left_index=True,
                              right_index=True)

        # if not using dynamic algorithm to determine when a
        # document mentions a specific topic:
        elif max_topics == 1:

            # assign highest weighted topic to each text record
            output['topic'] = [list(el).index(max(el)) for el in self.doc_topic_weights]

            # return highest weighted topic's weight
            output['weight'] = [max(el) for el in self.doc_topic_weights]

        else:
            # the following dictionary and list will be used to create a dataframe
            interim = {'topic': [], 'weight': [], 'priority': []}
            row_idx = []

            # iterate through each documents' topic weights
            for i in range(rows):
                doc_weights = self.doc_topic_weights.loc[i]
                temp = []

                # iterate through each topic weight
                for j in range(columns):
                    # create a list of topic, document-topic weight pairs
                    temp.append((j, doc_weights[j]))

                # select top weighted topics
                temp = sorted(temp, key=lambda el: el[1], reverse=True)[:max_topics]
                for j in range(max_topics):
                    interim['topic'].append(temp[j][0])
                    interim['weight'].append(temp[j][1])
                    interim['priority'].append(j + 1)  # Rank of topic relevance
                    row_idx.append(i)  # Equal to input text's index in output df

            output = pd.merge(output, pd.DataFrame(interim, row_idx),
                              left_index=True,
                              right_index=True)

        # get list of word-weight dictionaries for each topic, containing
        # heighest weighted terms. list index corresponds to topic index
        keyw = self.get_topic_terms(key_words)

        if key_words > 0:
            # Add top weighted topic terms to output
            output['topic_keyw'] = output['topic'].apply(lambda x: ', '.join(keyw[x].keys()))

        if cut_off is not None:
            # determine documents whose weights are too low for any topic
            dnt = output[output['weight'] < cut_off].copy()

            temp = pd.DataFrame(dnt.index.value_counts())
            temp = temp[temp[0] == self.decomposer.n_components]

            dnt = pd.merge(dnt, temp, left_index=True, right_index=True)

            # pass to class attribute
            self.docs_unclassified = dnt

            # update output
            output = output[output['weight'] >= cut_off]

        return output

    def get_feature_names(self):

        """
            Returns list of distinct terms found across all texts
        """

        if self.turn_method_off:
            return None

        return self.tfidf.get_feature_names()

    def get_topic_terms(self, n_words=10, coincide=False,
                        coincide_values='weights'):

        """
            Returns list of 'term : weight' dictionaries for each topic,
            containing top 'n_words' weighted terms. list index corresponds
            to topic index.

            coincide: bool. If True, terms present in one topic but missing
            in another topic will be added to the missing topic.

            coincide_values: str or int. When coincide is True. Determines
            what values to be assigned to the additional terms. Default
            'weights' value returns topic-term weight
        """

        output = []

        # get all topic term weights - store them in topic_term_weights
        # attribute if not already called
        if self.topic_term_weights is None:
            self.topic_term_weights = self.all_topic_term_weights()

        all_topic_term_weights = self.topic_term_weights

        if n_words == 0:
            return None

        # loop throught list of topic-term weight dictionaries, slicing them
        # to top n_words
        output = [{k: v for k, v in list(el.items())[:n_words]}
                  for el in all_topic_term_weights]

        if coincide:

            # all unique terms from output
            top_terms_all_topics = set([i for j in output for i in j.keys()])

            # loop through all topics
            for i in range(len(output)):

                # filling in term weights - dependent on coincide_values
                if coincide_values == 'weights':

                    # Constructing from all_topic_term_weights ensures
                    # dictionary keys remain in the same order for each topic
                    temp = {key: value for (key, value)
                            in all_topic_term_weights[i].items()
                            if key in top_terms_all_topics}

                    # overwrite output[i]
                    output[i] = temp

                elif coincide_values in [0, 'zero', 'zeros', 'zeroes']:

                    # missing terms for topic i
                    missing_terms = [el for el in top_terms_all_topics
                                     if el not in output[i].keys()]

                    # fill missing terms' weights as 0
                    temp = {key: (value if key not in missing_terms else 0)
                            for (key, value) in all_topic_term_weights[i].items()
                            if key in top_terms_all_topics}

                    # overwrite output[i]
                    output[i] = temp

                else:

                    raise ValueError('coincide_values argument invalid')

        return output

    def construct_topic_network(self, similarity_matrix, text_classify_df,
                                colour_map='Oranges', streamlit=False, similarity_threshold=0.1, topic_selected=0):

        # callable used to assign values between 0 and 1 to colours of colour_map
        colour_scale = PlotlyColorMap(colour_map, transparency=True)

        n_docs, n_topics = self.doc_topic_weights.shape

        # selecting similarity scores between topics, using the upper triangle
        # indices of the similarity_matrix
        t_r = range(n_topics)
        key_indices = [(i, j) for i in t_r for j in t_r if i < j]

        flat_sim = [similarity_matrix.iloc[i, j] for i, j in key_indices]
        min_sim = min(flat_sim)
        max_sim = max(flat_sim)

        # Scale similarity scores to range 0-1. This will allow the full
        # range of colours from colour_scale to be used and ensure the line
        # colours will match with the colours scale shown on the topic network
        # plot

        scaler = MinMaxScaler().fit([[min_sim], [max_sim]])  # Initialize

        # create custom colour scale
        custom_scale = [float(scaler.transform([[min_sim]])),
                        float(scaler.transform([[max_sim]]))]

        custom_scale_labels = custom_scale

        custom_scale = [colour_scale.rgba(el) for el in custom_scale]

        # Calculating polar coordinates for topic nodes. Every node is
        # arranged on a circle's circumference with 2-units Euclidean distance
        # between them.

        radians = 2 * pi / n_topics  # Angle between topic n and topic n + 1
        radius = 1 / sin(0.5 * radians)

        nodes = []
        keyw = self.get_topic_terms() # keywords for node labels
        keywords = []

        for i in range(n_topics):
            # proportion of documents that reference this topic - used for node size
            doc_ref_pcent = len(text_classify_df[text_classify_df['topic'] == i]) / n_docs

            keywords.append([str(w) for w in keyw[i].keys()])

            nodes.append([radius * cos(radians * i), radius * sin(radians * i), doc_ref_pcent])

        nodes = pd.DataFrame(nodes, columns=['x', 'y', 'size'])
        keywords_formatted = [', <br>'.join(words) for words in keywords]

        fig = go.Figure(data=[go.Scatter(x=nodes['x'],
                                         y=nodes['y'],
                                         mode='markers',
                                         marker_size=100 * nodes['size'],
                                         marker_color='darkGray',
                                         text=['Topic ' + str(i) + '<br>Key words: ' + str(keywords_formatted[i]) + '<br>' for i in nodes.index],
                                         hovertemplate='<b>%{text}</b><br>' +
                                                       'Documents referencing topic: %{marker.size:.1f}%' +
                                                       '<extra></extra>',
                                         showlegend=False),

                              # Hidden Scatter Trace - used to create colour scale for lines:
                              go.Scatter(x=[0] * len(flat_sim),
                                         y=[0] * len(flat_sim),
                                         name='Similarity score',
                                         mode='markers',
                                         marker=dict(
                                             size=[0]*len(flat_sim),
                                             cmax=custom_scale_labels[1],
                                             cmin=custom_scale_labels[0],
                                             color=flat_sim,
                                             colorscale=custom_scale,
                                             showscale=True
                                         ),
                                         hoverinfo='skip')
                              ],
                        layout=go.Layout(height=600,
                                         width=600,
                                         title='Topic Similarity Network',
                                         plot_bgcolor='#0b2d39',
                                         xaxis_visible=False,
                                         xaxis_fixedrange=True,
                                         yaxis_visible=False,
                                         yaxis_fixedrange=True,
                                         legend_title_side='top')
                        )

        # Add lines between nodes with colour scale indicating topic similarity
        # and matching legend values from the hidden scatter produced above
        for i, j in key_indices:
            # retrieve x and y coordinates of relevant topic nodes
            temp = nodes.loc[[i, j], list('xy')].copy()

            # select equidistant points on the line that runs from one node
            # to the other - These points will be used for their hoverboxes

            temp.reset_index(drop=True, inplace=True)  # Index is now 0,1
            temp.index = temp.index * 10  # Index is now 0,10

            # Extend temp dataframe to 10 records, index 0,1,...,10. Records
            # 1 to 9 will be empty
            temp = temp.reindex(range(temp.index[-1] + 1))

            # Linear interpolation will fill empty records
            temp.interpolate('linear', inplace=True)

            # get similarity score between topic i and topic j
            ij_similarity = similarity_matrix.iloc[i, j]

            # scale similarity score
            ij_similarity = float(scaler.transform([[ij_similarity]]))

            opacity = 0.25
            if i == topic_selected or j == topic_selected:
                opacity = 1.0
            # for dynamic changes in streamlit
            if ij_similarity >= similarity_threshold:

                # get colour value for line
                colour = colour_scale.rgba(ij_similarity)

                # Add line
                fig.add_trace(go.Scatter(x=temp['x'],
                                         y=temp['y'],
                                         mode='lines',
                                         opacity=opacity,
                                         text=['Topic ' + str(i) + ' --- Topic ' + str(j)] * len(temp),
                                         customdata=['{:.3f}'.format(ij_similarity)] * len(temp),
                                         hovertemplate='<b>%{text}</b><br><br>' +
                                                       'Similarity: %{customdata}' +
                                                       '<extra></extra>',
                                         line_color=colour,
                                         showlegend=False))

        if streamlit:
            fig.update_layout(title=None,
                              margin_t=1,
                              margin_r=1,
                              margin_l=1,
                              template='seaborn')

        return fig

    def vis_doc_topic_weights(self, all_topics_dist=False, add_cuts=False):

        """
            Creates visual showing distribution of document weights for each
            topic
        """
        plot_points = pd.DataFrame(columns=['x', 'y'])

        if add_cuts:
            # Initialize DecisionTreeRegressor
            splitter = DecisionTreeRegressor(max_depth=1)

            cut_points = pd.DataFrame(columns=['x', 'y1', 'y2'])

        # iterate through topics
        for i in range(self.doc_topic_weights.shape[1]):

            column = self.doc_topic_weights.iloc[:, i]

            plot_points = plot_points.append(pd.DataFrame(column).rename(columns={i: 'x'}),
                                             ignore_index=True)

            plot_points.fillna(i, inplace=True)

            if add_cuts:
                # attempt to find optimal topic weight threshold
                splitter.fit(pd.DataFrame(column), column)

                # Add threshold value to cut_points
                cut_points.loc[i] = [splitter.tree_.threshold[0],
                                     i - 0.5,
                                     i + 0.5]

        # build scatter of document weight distribution for each topic index
        fig = px.scatter(plot_points, 'x', 'y',
                         labels={'x': 'Document weight',
                                 'y': 'topic index'},
                         title='Document Weights per Topic')

        # Add an additional distribution above all topics to show all
        # document weights
        if all_topics_dist:

            i += 1
            plot_points['y'] = i

            fig.add_trace(go.Scatter(x=plot_points['x'],
                                     y=plot_points['y'],
                                     name='All Topic Weights',
                                     mode='markers',
                                     marker_color='Green'))

            if add_cuts:
                splitter.fit(plot_points[['x']], plot_points['x'])
                cut_points.loc[i] = [splitter.tree_.threshold[0],
                                     i - 0.5,
                                     i + 0.5]

        # add cut lines to visual
        if add_cuts:
            for i, row in cut_points.iterrows():
                fig.add_trace(go.Scatter(x=[row['x'], row['x']],
                                         y=[row['y1'], row['y2']],
                                         mode='lines',
                                         line_color='Red',
                                         showlegend=False))

        return fig

    def vis_doc_topic_var(self):

        df = pd.DataFrame(self.doc_topic_weights.var(1, ddof=0)
                          , columns=['variance'])

        df['dominant_topic_weight'] = self.doc_topic_weights.max(1)

        fig = px.scatter(df, 'variance', 'dominant_topic_weight')

        return fig

    def get_embedding_measure(self, n_words=20, mod='spacy'):

        output = []

        if mod == 'spacy':

            # load pretrained spacy model
            embedding_model = spacy.load("en_core_web_md")

            # get topic model
            topic_model = self.get_topic_terms(n_words)

            # loop through topics and get vector for each set of words
            for topic in topic_model:
                topic_embeddings = []
                for key, value in topic.items():

                    ek = embedding_model(key).vector
                    if ek is not None:
                        topic_embeddings.append(ek)
                        print(key)

                topic_average = np.mean(topic_embeddings, axis=0)
                output.append(topic_average)




        # use glove embedding
        elif mod == 'glove':

            # create empty dictionary to hold the embeddings
            glove_embeddings = {}

            # read embeddings from txt file in to dictionary
            print("*** Reading Glove file ***")
            with open("./models/glove.42B.300d.txt", 'r', encoding="utf-8") as embedding_txt:
                for line in embedding_txt:
                    values = line.split()
                    word = values[0]
                    vector = np.asarray(values[1:], "float32")
                    glove_embeddings[word] = vector

            print("*** Glove file read successfully ***")
            # get our topics
            topic_model = self.get_topic_terms(n_words)

            # loop through topics and get vector for each set of words
            for topic in topic_model:
                topic_embeddings = []
                for key, value in topic.items():

                    if key in glove_embeddings:
                        ek = glove_embeddings[key]
                        topic_embeddings.append(ek)
                topic_average = np.mean(topic_embeddings, axis=0)

                output.append(topic_average)





        else:
            output = None

        self.topic_embedding_similarity = output

    def post_processing_mode(self):
        """
            Empties tfidf and decomposer. Sets turn_method_off to true.

            Reducing size of the class, for streamlit app
        """
        self.tfidf = None
        self.decomposer = None
        self.turn_method_off = True

        return None


# compares 2 groups of topics pairwise and returns a distance matrix
def compare_topics(topic1, topic2=None):
    # if a second topic model isnt given then compare with itself (within topic comparisons)
    if not topic2:
        topic2 = topic1

    return (pd.DataFrame(
        [[float(cosine_similarity([i], [j])) for c, i in enumerate(topic1)] for k, j in enumerate(topic2)]))
