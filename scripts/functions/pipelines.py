# -*- coding: utf-8 -*-
"""
Created on Tue 20/05/2022

@author: Joe HC

Abstract pipelines and pipelines for specific packages can be found here.

Using the pipelines developed here:
    - Models can be iteratively trained to find optimal hyperparameters 
    - Different modelling algorithms can be compared to one another.

As this package uses algorithms from different python libraries, multiple 
pipelines are available.
"""

import logging
from typing import Optional, Any, Union
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from copy import deepcopy

from .descriptor_modifier import BaseDescriptorModifier
from .child_argument_handlers import IsInstanceArgHandler

from .spacy_utils import SpacyExtendedPipeline, SpacyGensimBOW

from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim.interfaces import TransformationABC as BaseGensimModel
from gensim.models.coherencemodel import CoherenceModel

import pandas as pd

# Initialize Logger
logging.getLogger(__name__)

######################
### Building custom descriptor modifier for topic modelling pipelines.
### update_object method unnecessary for this class.
class PipelineDescriptorModifier(BaseDescriptorModifier):
    def update_object(self, obj) -> None:
        return None


######################
### Building Abstract Topic Modelling Pipeline
### used as a reference class for aguments in Pipeline 
### Component Appliers.

class AbstractModellingPipeline(ABC):
    """
    This pipeline is used to combine components of: 
        - text preprocessing
        - word vectorization
        - topic modelling
        - topic assessment
    """
    component_order = (
        'preprocessing',
        'vectorization',
        'model'
    )

######################
### Building classes to update and apply a specific 
### component in the topic modelling pipelines

class AbstractPipelineComponentApplier(ABC):

    def __init__(
        self, 
        *args, 
        **kwargs
    ):
        pass
    
    @abstractmethod
    def update_component(
        self, 
        obj: AbstractModellingPipeline, 
        component: str,
        updates: dict[str, Any]
    ) -> None:
        pass

    @abstractmethod
    def apply_component(
        self, 
        obj: AbstractModellingPipeline, 
        component: str,
        input: Any
    ):
        pass

    def __call__(
        self,
        obj: AbstractModellingPipeline, 
        component: str,
        updates: dict[str, Any], 
        input: Any
    ) -> Any:
        self.update_component(obj, component, updates)
        return self.apply_component(obj, component, input)


class DefaultPipelineComponentApplier(AbstractPipelineComponentApplier):
    _key = type(None) # default key
    
    def update_component(
        self, 
        obj: AbstractModellingPipeline, 
        component: str,
        updates: dict[str, Any]
    ):
        for k, v in updates.items():
            setattr(getattr(obj, component), k, v)
    
    def apply_component(
        self, 
        obj: AbstractModellingPipeline, 
        component: str,
        input: Any
    ):
        return getattr(obj, component)(input)


######################
### ArgHolder and Updater classes: These are abstactive layers, 
### that collect all new hyperparameters for each component in the 
### topic modelling pipelines. The updater class is then passed to 
### the topic modelling class to update the pipeline components.

class ArgHolder:
    """
    Blank class, used to hold pipeline component arguments to be updated.
    """
    pass


class UpdateArgs:
    """
    Holds arguments to update for each step in the topic modelling pipeline.
    """
    def __init__(
        self, 
        pipeline: AbstractModellingPipeline
    ):
        """
        Extracts key component steps of the pipeline using component_order.
        """
        for step in pipeline.component_order:
            setattr(self, step, ArgHolder())

######################
### Building Topic Modelling Pipelines

# Base pipeline class
class BaseModellingPipeline(AbstractModellingPipeline):
    """
    This pipeline is used to combine components of: 
        - text preprocessing
        - word vectorization
        - topic modelling
        - topic assessment
    """

    # Descriptor Classes
    preprocessing = PipelineDescriptorModifier(
        base_class = AbstractPipelineComponentApplier, 
        argument_type_handling = IsInstanceArgHandler(), 
        error_message = 'preprocessing argument not recognised'
    )

    vectorization = PipelineDescriptorModifier(
        base_class = AbstractPipelineComponentApplier, 
        argument_type_handling = IsInstanceArgHandler(), 
        error_message = 'vectorization argument not recognised'
    )

    model = PipelineDescriptorModifier(
        base_class = AbstractPipelineComponentApplier, 
        argument_type_handling = IsInstanceArgHandler(), 
        error_message = 'model argument not recognised'
    )

    def __init__(
        self,
        preprocessing: SpacyExtendedPipeline,
        vectorization,
        model
    ):
        self.preprocessing = preprocessing
        self.vectorization = vectorization
        self.model = model

    def apply_pipeline_partial(
        self, 
        component_step: str,
        inputs: dict[str, Any], 
        updates: Optional[UpdateArgs] = None,
    ) -> dict[str, Union[pd.Series, NDArray]]:
        """
        Applies each step of the model to input(s) data.

        Able to partially apply the pipeline to from a specific component_step onwards.
        """
        # Check component_step argument passed is legitimate
        assert component_step in self.component_order, f'component_step: {component_step}, not recognised as part of pipeline.'

        # initialize UpdateArgs if None passed to updates
        if updates is None:
            updates = UpdateArgs(self)

        # Initialize outputs dictionary
        outputs = dict()

        # run through the pipeline using component order as reference 
        # starting from component_step
        next_input = inputs.get(component_step)
        component_idx = self.component_order.index(component_step)
        for step in self.component_order[component_idx:]: 
            logging.info(f'Applying {step} step from pipeline.')

            outputs[step] = getattr(self, f'_{step}_modifier')(
                obj = self,
                component = f'_{step}', # private name
                updates = getattr(updates, step).__dict__, 
                input = next_input
            )

            # update next_input for next step
            next_input = outputs.get(step)

        return outputs

    def apply_pipeline(
        self, 
        texts: Union[pd.Series, NDArray],
        updates: Optional[UpdateArgs] = None
    ) -> dict[str, Union[pd.Series, NDArray]]:

        component_step = self.component_order[0]
        inputs = {component_step: texts}

        return self.apply_pipeline_partial(
            component_step = component_step, 
            inputs = inputs,
            updates = updates
        )

    @abstractmethod
    def get_score_dict(
        self
    ) -> dict[str, Union[int, float]]:
        pass

######################
### Building Gensim Topic Modelling Pipeline:


### Gensim models must have their initialization arguments recorded
def gensim_model_holder(
    cls: BaseGensimModel, 
    **init_kwargs
):
    instance = cls(
        id2word = init_kwargs.pop(
            'id2word', 
            Dictionary([['Placeholder']])
        ),
        **init_kwargs
    )

    init_kwargs['id2word'] = instance.id2word

    setattr(instance, 'init_kwargs', init_kwargs)

    return instance


### Gensim needs it's own unique pipeline component applier
class GensimPipelineComponentApplier(AbstractPipelineComponentApplier):
    _key = BaseGensimModel

    def update_component(self) -> None:
        pass

    def apply_component(self) -> None:
        pass

    def __call__(
        self, 
        obj: AbstractModellingPipeline, 
        component: str,
        updates: dict[str, Any],
        input: Any
    ):

        gensim_component = deepcopy(getattr(obj, component))

        # retrieve init_kwargs attribute
        init_kwargs = getattr(gensim_component, 'init_kwargs')

        # set initialization arguments
        reinit_kwargs = init_kwargs.copy()
        reinit_kwargs.update(
            updates, 
            corpus = input, 
            id2word = obj.preprocessing.dictionary
        )

        # reinitialize model
        gensim_component.__init__(**reinit_kwargs)
        setattr(gensim_component, 'init_kwargs', init_kwargs)

        # update component in obj
        setattr(obj, component, gensim_component)

        return gensim_component[input]


### Gensim Pipelines
class GensimPipeline(BaseModellingPipeline):

    def __init__(
        self,
        preprocessing: SpacyGensimBOW,
        vectorization: TfidfModel,
        model: BaseGensimModel,
        coherence_kwargs: Union[dict, list[dict], None] = None
    ):
        """
        coherence_kwargs is a (list of) dictionaries with keys for gensim coherence parameter names:
            - window_size
            - coherence
        """
        super().__init__(
            preprocessing = preprocessing,
            vectorization = vectorization,
            model = model
        )
        # Process argument passed to coherence_kwargs
        if coherence_kwargs:
            if type(coherence_kwargs) is dict:
                self.coherence_kwargs = [coherence_kwargs]
            else:
                self.coherence_kwargs = coherence_kwargs
        else: self.coherence_kwargs = [dict()] # default coherence values


    def get_score_dict(
        self,
        inputs: dict[str, Any]
    ) -> dict[str, Union[int, float]]:
        """
        Coherence measures applied here, uses Gensim CoherenceModel as a base.

        Output is a dictionary of coherence scores for multiple CoherenceModel parameters.
        """
        output = {}
        
        for kwargs in self.coherence_kwargs:
            logging.info(f'Measuring topic coherence with parameters: {kwargs}')
            coherence_instance = CoherenceModel(
                model = self.model,
                texts = self.preprocessing.texts, 
                corpus = inputs.get('model'), 
                dictionary = self.preprocessing.dictionary,
                processes = 1, 
                **kwargs
            )
            
            # Construct output key
            output_key = coherence_instance.coherence
            output_key += '__ws_' + str(coherence_instance.window_size)

            # Add coherence score to output dictionary
            output[output_key] = coherence_instance.get_coherence()

            del coherence_instance

        return output