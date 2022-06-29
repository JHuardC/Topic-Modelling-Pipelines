# -*- coding: utf-8 -*-
"""
Created on Tue 12/04/2022

@author: Joe HC

Using the pipelines developed pipelines.py the TuningPipeline class can:
    - Iteratively train a model to find optimal hyperparameters 
    - Compare different modelling algorithms.
"""

import logging
from typing import Optional, Any
import itertools as itools
from copy import deepcopy

from .pipelines import BaseModellingPipeline, UpdateArgs

import pandas as pd

# Initialize Logger
logging.getLogger(__name__)


######################

# Hyperparameter tuning class
class TuningPipeline:
    """
    This pipeline is used to combine components of: 
        - Children of the BaseModellingPipeline
        - feature extraction
    
    These pipelines support hyperparameter tuning.
    """

    def __init__(
        self,
        topic_pipeline: BaseModellingPipeline,
        hyperparameter_grid: Optional[dict[str, list[Any]]] = None
    ):
        self.topic_pipeline = topic_pipeline
        self.hyperparameter_grid = hyperparameter_grid

    def sort_grid_key(
        self,
        key_value: tuple[str, Any]
    ) -> list[tuple[str, list[Any]]]:
        """
        Optimizing hyperparameter order for grid search.

        Given a key_value to update a pipeline's component's 
        hyperparameter with, this method finds and returns the components 
        position number as a step to be applied in the pipeline.

        Note, the order of components to change in hyperparameter grid should 
        be the reverse of the order the components are called in the 
        Pipeline, as specified by class variable component_order on 
        the topic_pipeline object.
        """
        key, _ = key_value
        component_referenced = key.split('__')[0]

        return self.topic_pipeline.component_order.index(component_referenced)

    def ordered_cartesian_product(
        self, 
        hyperparameter_grid: dict[str, list[Any]]
    ) -> itools.product:

        # cartesian products of hyperparameter grid
        ordered_parameter_permutations = sorted(
            hyperparameter_grid.items(),
            key = self.sort_grid_key
        )
        ordered_parameter_permutations = [
            itools.zip_longest([], v, fillvalue = k) 
            for k, v in ordered_parameter_permutations
        ]
        ordered_parameter_permutations = itools.product(
            *ordered_parameter_permutations
        )

        return ordered_parameter_permutations


    @staticmethod
    def pass_parameters(
        topic_pipeline: BaseModellingPipeline,
        update_parameters: list[tuple[str, Any]]
    ) -> UpdateArgs:
        """
        Generates UpdateArgs class from update_parameters, to be used
        by a BaseModellingPipeline to update it's model parameters.
        """
        updater = UpdateArgs(topic_pipeline)

        # run through parameters to update
        for parameter in update_parameters:
            attr_keys, value = parameter

            # Chain down nested hyperparameters in topic_model
            attr_chain = []
            obj = topic_pipeline
            for attr_str in attr_keys.split('__'):
                attr_chain.append((obj, attr_str))
                obj = deepcopy(getattr(obj, attr_str))

            # Modify attr_chain list to pass new hyperparameter value
            # to Updater class instead of passing back to topic_pipeline.
            modified_chain = [
                (updater, attr_keys.split('__')[0]), 
                (getattr(updater, attr_keys.split('__')[0]), attr_keys.split('__')[1])
            ]

            attr_chain = modified_chain + attr_chain[2:]
            
            # Chain up: Setting new values for nested hyperparameters
            for attr_obj, attr_str in reversed(attr_chain):
                setattr(attr_obj, attr_str, value)
                value = attr_obj
            updater = value

        return updater

    
    def gridsearch(
        self,
        texts: pd.Series,
        hyperparameter_grid: Optional[dict[str, list[Any]]] = None
    ) -> list[dict[str, Any]]:
        """
        Grid search hyperparameter tuning. 
        Systematically searches through hyperparameter 
        permutations and records performance using assessment 
        module.
        """
        # If hyperparameter_grid is not specified then use 
        # instance attributes passed on initialization.
        if not hyperparameter_grid:
            hyperparameter_grid = self.hyperparameter_grid
        # cartesian products of hyperparameter grid
        hyperparameter_grid = self.ordered_cartesian_product(
            hyperparameter_grid
        )

        # Score records for each hyperparameter permutation
        score_records = []

        # Copy processing pipeline
        topic_pipeline = deepcopy(self.topic_pipeline)

        # inputs for each component in the processing pipeline
        # allows for partial pipeline implementation
        inputs = dict(
            itools.zip_longest(
                topic_pipeline.component_order, 
                [texts], 
                fillvalue = None
            )
        )

        # Run through each hyperparameter permutation
        previous_parameters = set()
        for i, parameters in enumerate(hyperparameter_grid):

            logging.info(f'Applying to pipeline the parameters: {parameters}.')

            # find the components that will change from the last permutation
            parameter_dif = [
                el for el in parameters 
                if el not in previous_parameters
            ]

            logging.info(f'Applying to pipeline the new parameters: {parameter_dif}.')
            
            # find earliest component in the pipeline that 
            # this parameter permutation is altering
            initial_component = parameter_dif[0][0].split('__')[0]

            # generate UpdateArgs class to be passed to topic_pipeline
            updater = self.pass_parameters(
                topic_pipeline,
                parameter_dif
            )

            pipeline_outputs = topic_pipeline.apply_pipeline_partial(
                component_step = initial_component if i else topic_pipeline.component_order[0],
                inputs = inputs,
                updates = updater
            ) 

            # delete final output in pipeline_outputs as it is unnecessary here
            del pipeline_outputs[topic_pipeline.component_order[-1]] 

            # pipeline outputs are now used to update inputs dictionary
            for component, output in pipeline_outputs.items():

                # get current component index
                next_component = topic_pipeline.component_order.index(
                    component
                )
                # increment to next index value
                next_component += 1 
                # retrieve next component value
                next_component = topic_pipeline.component_order[next_component]

                inputs[next_component] = output

            logging.info("Get scores for current parameters")
            score_dict = topic_pipeline.get_score_dict(inputs)
            param_dict = {k: v for k, v in parameters}
            param_dict.update(**score_dict)
            score_records.append(param_dict)
            
            previous_parameters = {*parameters}

        self.score_records = score_records

        return score_records