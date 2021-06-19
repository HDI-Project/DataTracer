# -*- coding: utf-8 -*-

"""DataTracer core module.

This module defines the DataTracer class.
"""

import json
import os
import pickle

from mlblocks import MLPipeline

PRETRAINED_DIR = os.path.join(os.path.dirname(__file__), 'pretrained')
PIPELINE_DIR = os.path.join(os.path.dirname(__file__), 'jsons/pipelines')
PRIMITIVE_DIR = os.path.join(os.path.dirname(__file__), 'jsons/primitives')


class DataTracer:
    """DataTracer Class.

    The DataTracer Class provides a unified and standardized access to
    all the Data Lineage functionalities of the project.

    Args:
        pipeline (str, dict or MLPipeline):
            Pipeline to use. It can be passed as:
                * An ``str`` with a path to a JSON file.
                * An ``str`` with the name of a registered pipeline.
                * An ``MLPipeline`` instance.
                * A ``dict`` with an ``MLPipeline`` specification.
        hyperparameters (dict):
            Additional hyperparameters to set to the Pipeline.
    """

    def _get_mlpipeline(self):
        pipeline = self._pipeline
        if isinstance(pipeline, str):
            if os.path.isfile(pipeline):
                with open(pipeline) as json_file:
                    pipeline = json.load(json_file)
            elif os.path.isfile(os.path.join(PIPELINE_DIR, pipeline + '.json')):
                with open(os.path.join(PIPELINE_DIR, pipeline + '.json')) as json_file:
                    pipeline = json.load(json_file)

        if isinstance(pipeline, dict):
            if 'primitives' in pipeline:
                for idx in range(len(pipeline['primitives'])):
                    primitive = pipeline['primitives'][idx]
                    if isinstance(primitive, str):
                        if os.path.isfile(primitive):
                            with open(primitive) as json_file:
                                primitive = json.load(json_file)
                        elif os.path.isfile(os.path.join(PRIMITIVE_DIR, primitive + '.json')):
                            with open(os.path.join(PRIMITIVE_DIR, primitive + '.json'))\
                                    as json_file:
                                primitive = json.load(json_file)
                    pipeline['primitives'][idx] = primitive

        mlpipeline = MLPipeline(pipeline)
        if self._hyperparameters:
            mlpipeline.set_hyperparameters(self._hyperparameters)

        return mlpipeline

    def __init__(self, pipeline, hyperparameters=None):
        self._pipeline = pipeline
        self._hyperparameters = hyperparameters
        self._mlpipeline = self._get_mlpipeline()

    def fit(self, datasets):
        """Fit the pipeline to the given data.

        Args:
            datasets (dict):
                Dict mapping dataset names to tuples containing a MetaData
                instance and a dict with the tables of the dataset loaded
                as DataFrames.
        """
        self._mlpipeline = self._get_mlpipeline()
        self._mlpipeline.fit(dict_of_databases=datasets, tables={})

    def solve(self, tables, **kwargs):
        """Solve the data lineage problem.

        The underlaying pipeline is executed and the outputs are returned.

        Args:
            tables (dict):
                Dictionary of tables from a dataset loaded as DataFrames.
            **kwargs:
                Any additional keyword arguments are passed down to the
                pipeline.

        Returns:
            object:
                This method returns the outputs returned by the pipeline.
        """
        return self._mlpipeline.predict(tables=tables, **kwargs)

    def save(self, path: str):
        """Save this object using pickle for later usage.

        Args:
            path (str):
                Path to the file where the serialization of
                this object will be stored.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)

    @classmethod
    def load(cls, datatracer: str):
        """Load a DataTracer instance from a pickle file.

        Args:
            datatracer (str):
                Name of the datatracer to load, or path to a previously saved
                instance.

        Returns:
            DataTracer

        Raises:
            ValueError:
                If the serialized object is not a DataTracer instance.
        """
        if os.path.isfile(datatracer):
            path = datatracer
        else:
            path = os.path.join(PRETRAINED_DIR, datatracer + '.dt')
            if not os.path.isfile(path):
                raise ValueError('Unknown datatracer: {}'.format(datatracer))

        with open(path, 'rb') as pickle_file:
            datatracer = pickle.load(pickle_file)
            if not isinstance(datatracer, cls):
                raise ValueError('Serialized object is not a DataTracer instance')

            return datatracer
