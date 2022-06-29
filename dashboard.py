# -*- coding: utf-8 -*-
"""
Created on Wed Sep 1 2021
@author: Joe L/Joe HC/Steven H
The main file for this project that will output the results of the model into a dashboard.
"""

import pandas as pd
from scripts.functions.model import compare_topics
from scripts.functions.utils import date_returner

import plotly.express as px
import plotly.graph_objects as go
import pickle
import streamlit as st
from collections import Counter
import datetime

st.set_page_config(layout='wide')


########################################
############# load pickles #############

# In order to make use of streamlit's cache function, the data should be
# loaded in via a custom function

@st.cache
def load():
    output = []

    pickle_dir = r'./data/pickles/{}.pkl'

    petition_topics = pd.read_pickle(pickle_dir.format(r'petition_topics'))
    petition_topics.dates = pd.to_datetime(petition_topics.dates)

    output.append(petition_topics)

    with open(pickle_dir.format(r'model_set'), 'rb') as infile:
        output.append(pickle.load(infile))
    infile.close()

    return output


petition_topics, model = load()


############################################
############# Custom Functions #############

# Get unchanging variables to be stored in cache through custom functions

def get_top_weighted_terms(modeller, n_words):
    # This function is built to take the top n weighted terms for each topic
    # from TopicDecompositionSklearn attribute topic_term_weights. This
    # attribute contains dictionairies already ordered by weights

    output = [{k: v for k, v in list(el.items())[:n_words]}
              for el in modeller.topic_term_weights]

    return output


@st.cache(allow_output_mutation=True)
def get_aggreated_topic_references_visual(fit_transform_classify_output, min_date, max_date):
    ftco = fit_transform_classify_output  # aliasing

    # group ftco by topic and/or priority: Note that priority might not be
    # a column in ftco
    grouping_columns = [el for el in ['topic', 'priority'] if el in ftco.columns]
    ftco = ftco[(ftco['dates'].dt.date >= min_date) & (ftco['dates'].dt.date <= max_date)]

    vis_data = ftco.groupby(grouping_columns, as_index=False)['text'].agg({'count': 'count'})

    vis_data[grouping_columns] = vis_data[grouping_columns].astype(str)

    fig = px.bar(vis_data,
                 x='count',
                 y='topic',
                 color='priority',
                 orientation='h',
                 template='seaborn')

    fig.update_layout(yaxis_title='Topic Index',
                      xaxis_title='Number of Documents',
                      yaxis_fixedrange=True,
                      xaxis_fixedrange=True,
                      xaxis_side='top',
                      yaxis_autorange='reversed',
                      margin_t=1,
                      margin_r=1,
                      margin_l=1)

    return fig


def get_timeseries(fit_transform_classify_output, chosen_topic):
    ftco = fit_transform_classify_output  # aliasing

    # group ftco by topic and/or priority: Note that priority might not be
    # a column in ftco
    grouping_columns = [el for el in ['month'] if el in ftco.columns]

    vis_data = ftco[['text', 'dates', 'topic']].reset_index()
    vis_data['month'] = vis_data.apply(lambda row: date_returner(row['dates']), axis=1)
    vis_data = vis_data[vis_data['topic'] == chosen_topic]

    vis_data_count = vis_data.groupby(['month', 'topic']).text.count().reset_index()

    series = px.bar(vis_data_count, x='month', y='text', template='seaborn')

    series.update_layout(xaxis_title='Month',
                         yaxis_title='Number of Documents',
                         width=2000)

    return series

def emerging_topics(fit_transform_classify_output):
    ftco = fit_transform_classify_output  # aliasing

    # group ftco by topic and/or priority: Note that priority might not be
    # a column in ftco
    vis_data = ftco[['text', 'dates', 'topic']].reset_index()
    vis_data['month'] = vis_data.apply(lambda row: date_returner(row['dates']), axis=1)
    monthly = pd.crosstab(vis_data['topic'], vis_data['month'], vis_data['text'], aggfunc='count')
    monthly.fillna(0, inplace=True)

    monthly.reset_index(inplace=True)

    keep_cols = ['topic']

    # calculate emergence for each month
    for i in range(3, len(monthly.columns)):
        col = monthly.columns[i]
        months_earlier = monthly.columns[i-2]
        str_col = str(col)
        monthly[str_col+'_emergence'] = monthly.apply(lambda row: get_emergence(row[months_earlier], row[col]), axis=1)
        monthly[str_col+'_emergence_rounded'] = monthly.apply(lambda row: round(row[str_col+'_emergence'], 2), axis=1)
        keep = str_col+'_emergence_rounded'
        keep_cols.append(keep)

    monthly_scores = pd.DataFrame(monthly[keep_cols])

    for col in monthly_scores.columns:
        monthly_scores.rename(columns={
            col : col.replace('_emergence_rounded', '')
        }, inplace=True)

    monthly_scores = monthly_scores.melt(id_vars='topic', value_name='emergence_score', var_name='month')

    fig = go.Figure(data=go.Heatmap(
        z=monthly_scores['emergence_score'],
        x=monthly_scores['month'],
        y=monthly_scores['topic'],
        hoverongaps=False,
        colorscale='Oranges'
        ))
    fig.update_layout(width=1500, height=800)

    return fig

def get_emergence(start, end):
    if start + end == 0:
        return 0
    else:
        return (end - start) / (start + end)


def label_topics(topic_emergence):
    if topic_emergence > 5:
        return "emerging"
    if topic_emergence < -3:
        return "receding"
    else:
        return "stagnant"


@st.cache
def get_topic_network(modeller=model, ftco=petition_topics, threshold=0.01, selected_topic=0, shared_length_set=600):
    # order topic-terms by term
    term_weights = [[v for k, v in sorted(el.items(), key=lambda x: x[0])]
                    for el in modeller.topic_term_weights]
    fig = modeller.construct_topic_network(compare_topics(term_weights),
                                           ftco,
                                           streamlit=True, similarity_threshold=threshold,
                                           topic_selected=selected_topic)

    height = shared_length_set

    if shared_length_set > 600:
        height = 600

    fig.update_layout(height=height)

    return fig


#####################################
############# Streamlit #############

"""# Web petition topic modelling"""

""" You can view some exploratory analysis on the data by expanding the box below: """

ta_expander = st.expander('View report text analysis.')

with ta_expander:
    left_ta, centre_ta, right_ta = st.columns(3)

    with left_ta:
        st.write("The most common nouns in the dataset are:")
        for noun in model.top_nouns:
            st.write(noun[0] + " mentioned: " + str(noun[1]) + " times.")

    with centre_ta:
        st.write("The most commonly used words according to their weighting throughout all the documents are:")
        for word in model.top_scored_words:
            st.write(word)

    with right_ta:
        st.write("""You can view the most common words across all topics below. The list below shows how many times a 
        word is one of the most important words in a topic: """)

        num_words = st.slider(label='Most important number of words: ', min_value=4, max_value=30)

        weighted_words_scored = model.get_topic_terms(n_words=num_words)
        weighted_words = [str(k) for d in weighted_words_scored for k in d.keys()]
        words_counted = Counter(weighted_words)
        top_weighted_words = words_counted.most_common(10)

        for word in top_weighted_words:
            st.write(word[0] + " was one of the " + str(num_words) + " most important words in: " + str(
                word[1]) + " topics.")



left_ma, centre_ma, right_ma = st.columns([1, 1, 1])

# Plot
with left_ma:
    """
    ## Topic reference guide
    The below lists have the top ten words for each topic so you can easily refer to them later:
    """

    reference_words = model.get_topic_terms(n_words=10)
    reference_lists = []
    for d in reference_words:
        reference_list = list(d.keys())
        reference_lists.append(reference_list)

    topics = model.doc_topic_weights.shape[1]

    for topic in range(0, topics):
        st.text('Topic ' + str(topic) + ': ' + str([word for word in reference_lists[topic][0:5]]), )

with centre_ma:
    st.header('Number of documents by topic and referential priority (a)')
    st.write("""You can filter the visual below by date to view only the documents from a certain date range. 
    This range is inclusive.""")
    st.caption("(a) Where referential priority is defined as how strongly related a document is topic.")

    min_date = st.date_input('Select the minimum date you would like to include:', value=datetime.date(2020, 1, 1))
    max_date = st.date_input('Select the maximum date you would like to include:', value=datetime.date(2021, 9, 1))

    fig = get_aggreated_topic_references_visual(petition_topics, min_date=min_date, max_date=max_date)
    shared_length = 800

    # Update Height
    fig.update_layout(height=shared_length)

    # load visual to app
    st.plotly_chart(fig,
                    use_container_width=True,
                    config=dict(displayModeBar=False))

with right_ma:
    st.header('Topic network visual (a)')
    """
    Below you can view the relationships between topics. The colour of the lines joining each topic from around the 
    outside of the network indicates how related the topic is to the topic on the other end of the line is the higher 
    the number on the scale the more closely related topics are.
    """
    st.caption("(a) The topic selected by the slider on the below will have it's relationships highlighted below.")

    topic_n = st.slider('Topic selector',
                        min_value=0,
                        max_value=model.doc_topic_weights.shape[1] - 1,
                        value=7,
                        step=1,
                        format='%i')

    threshold = st.slider(label='Select the minimum similarity for the topics to be considered connected: ',
                          min_value=0.0, max_value=1.0, step=0.1, value=0.3)

    network = get_topic_network(threshold=threshold, selected_topic=topic_n, shared_length_set=shared_length)
    st.plotly_chart(network, use_container_width=True)

words_expander = st.expander('View more information on weighted keywords by topic.')

with words_expander:
    top_terms_title = st.empty()

    st.caption("(a) Where weighted keywords are the words most important to the given topic.")

    n_words = st.slider('Top weighted key words',
                        min_value=10,
                        max_value=50,
                        value=10,
                        step=5,
                        format='%i')

    # get top n_words for each topic
    topic_terms = model.get_topic_terms(n_words)

    # topic weight visuals title
    format_values = [str(topic_n), str(n_words)]
    title = 'Topic {}: Top {} weighted key words (a)'.format(*format_values)

    top_terms_title.header(title)

    # build bar chart
    f = px.bar(x=list(topic_terms[topic_n].values())[::-1],
               y=list(topic_terms[topic_n].keys())[::-1],
               orientation='h',
               template='seaborn')

    f.update_layout(yaxis_title=None,
                    xaxis_title='Weights',
                    yaxis_fixedrange=True,
                    xaxis_fixedrange=True,
                    xaxis_side='top',
                    showlegend=False,
                    margin_t=1,
                    margin_r=1,
                    margin_l=1)

    if 25 * model.doc_topic_weights.shape[1] - 200 < 35 * n_words:
        shared_length = 35 * n_words
    else:
        shared_length = 25 * model.doc_topic_weights.shape[1] - 200

    f.update_layout(height=shared_length)

    # load visual to app
    st.plotly_chart(f,
                    use_container_width=True,
                    config=dict(displayModeBar=False))

"""
## Number of documents by topic and date.
The time series below shows the number of documents where the topic selected was the top priority by date.
"""
topic_chosen = st.slider('Select a topic to view: ', min_value=0, max_value=model.doc_topic_weights.shape[1] - 1)
st.write(get_timeseries(petition_topics, chosen_topic=topic_chosen))

"""
## Topic emergence

The heatmap below shows the emergence of topics by month. Where emergence is defined as:

Documents Referencing Topic in Month - Documents Referencing Topic Three Months Ago / Documents Referencing Topic in 
Month + Documents Referencing Topic Three Months Ago
"""
st.write(emerging_topics(petition_topics))