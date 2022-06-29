# -*- coding: utf-8 -*-
"""
Created on Thur Apr 28 2022

@author: Joe HC

Classes to extract and hold key topic modelling  outputs.
"""
from dataclasses import dataclass
from abc import abstractmethod, ABC
from typing import Any, Union, Optional
import pandas as pd
from .pipelines import AbstractModellingPipeline

@dataclass
class TopicModelOutputs:
    """
    Standard data class, holds key features from topic models
    for use in dashboard.
    """
    feature_names: dict[int, str]
    doc_topic_weights: pd.DataFrame
    topic_term_weights: list[dict[str, float]]
    doc_topic_summary: Optional[pd.DataFrame]
    docs_unclassified: Optional[pd.DataFrame]


class AbstractTopicModelExtractor(ABC):
    @abstractmethod
    def __init__(
        self, 
        topic_model_outputs: dict[str, Any], 
        topic_model: AbstractModellingPipeline
    ):
        pass


class GensimTopicModelExtractor(AbstractTopicModelExtractor):
    """
    Extractor for Gensim based topic models.
    Components are extracted to a TopicModelOutputs class.
    """
    def __init__(
        self, 
        texts: Union[pd.Series, pd.DataFrame], 
        topic_model_outputs: dict[str, Any], 
        topic_model: AbstractModellingPipeline,
        summary_method: Union[str, dict[str, Union[int, float]]] = 'dynamic'
    ):

        """ texts: list-like entity of text data model was trained on.

            topic_model_outputs:

            topic_model:

            summary_method: doc_topic_summary dataframe is generated from arguments 
            passed to texts, topic_model_outputs and the topic_model parameters.

            max_topics: int or 'dynamic'. The maximum number of topics to
            assign to each documnet. Order of assignment follows document-topic
            weights.

            key_words: int. Return the top term-topic weighted terms for each
            topic assigned to a text.

            cut_off: float. Specifies the document-topic weighting threshold
            required to be returned. If None then the top 'max_topics' number
            of document-topic pairs will be returned for each document.
        """

        self.texts = texts

        self.feature_names = dict(topic_model.model.id2word)

        self.doc_topic_weights = pd.DataFrame(topic_model_outputs.get('model'))
        self.doc_topic_weights = self.doc_topic_weights.applymap(lambda el: el[-1])

        self.topic_term_weights = topic_model.model.show_topics(
            num_topics = topic_model.model.num_topics, 
            num_words = len(topic_model.model.id2word), 
            formatted = False
        )
        self.topic_term_weights = [
            dict(el[-1]) for el in self.topic_term_weights
        ]

    def get_data_class(self):
        return TopicModelOutputs(
            feature_names = self.feature_names, 
            doc_topic_weights = self.doc_topic_weights, 
            topic_term_weights = self.topic_term_weights, 
            doc_topic_summary = self.texts, 
            docs_unclassified = None
        )