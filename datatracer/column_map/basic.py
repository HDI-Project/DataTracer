import logging

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from datatracer.column_map.base import ColumnMapSolver
from datatracer.column_map.transformer import Transformer

LOGGER = logging.getLogger(__name__)


class BasicColumnMapSolver(ColumnMapSolver):
    """Basic Solver for the data lineage problem of column dependency."""

    def __init__(self, threshold=0.1, *args, **kwargs):
        self._model_args = args
        self._model_kwargs = kwargs
        self._threshold = threshold
        self._linear_weight_threshold = 1e-4
        self._linear_score_threshold = 0.95

    def _get_importances(self, X, y):
        model = RandomForestRegressor(*self._model_args, **self._model_kwargs)
        model.fit(X, y)

        return model.feature_importances_

    def _convert_linear_importances(self, weights):
        new_weights = (weights > self._linear_weight_threshold) / \
            sum(weights > self._linear_weight_threshold)

        return new_weights

    def solve(self, tables, foreign_keys, target_table, target_field):
        """Find the fields which contributed to the target_field the most.

        The output is a dictionary containing the fields that contributed the
        most to the given target field as keys, specified as a tuple containing
        both table name and field name, and the score obtained as values.

        Args:
            tables (dict):
                Dict containing table names as input and ``pandas.DataFrames``
                as values.
            foreign_keys (list):
                List of foreign key specifications.
            target_table (str):
                Name of the table that contains the target field.
            target_field (str):
                Name of the target field.

        Returns:
            dict:
                Dictionary of field specification tuples and scores.
        """
        transformer = Transformer(tables, foreign_keys)

        X, y = transformer.forward(target_table, target_field)
        if len(X.shape) != 2:  # invalid X shape
            return {}
        elif X.shape[0] == 0 or X.shape[1] == 0:  # empty dimension
            return {}

        try:
            reg = LinearRegression(fit_intercept=False).fit(X, y)
            if reg.score(X, y) > self._linear_score_threshold:
                importances = self._convert_linear_importances(reg.coef_)
            else:
                importances = self._get_importances(X, y)
        except BaseException:
            importances = self._get_importances(X, y)

        ret_dict = transformer.backward(importances)
        flag = True
        while flag:
            flag = False
            new_rets = ret_dict.copy()
            total_score = sum(ret_dict.values())
            for field, score in ret_dict.items():
                if score < total_score * self._threshold / len(ret_dict):
                    del new_rets[field]
                    flag = True
            ret_dict = new_rets
        return ret_dict
