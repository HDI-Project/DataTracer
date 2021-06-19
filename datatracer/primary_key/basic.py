"""Basic Primary Key Solver module."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from datatracer.primary_key.base import PrimaryKeySolver


class BasicPrimaryKeySolver(PrimaryKeySolver):

    def __init__(self, threshold=[i / 20 for i in range(20)], *args, **kwargs):
        self._model_args = args
        self._model_kwargs = kwargs
        self._threshold = threshold

    def _feature_vector(self, table, column_name):
        column = table[column_name]
        return [
            list(table.columns).index(column_name),
            0.0 if len(table.columns) == 0 else list(
                table.columns).index(column_name) / len(table.columns),
            1.0 if column.nunique() == len(column) else 0.0,
            0.0 if len(column) == 0 else column.nunique() / len(column),
            1.0 if "key" in column.name else 0.0,
            1.0 if "id" in column.name else 0.0,
            1.0 if "_key" in column.name else 0.0,
            1.0 if "_id" in column.name else 0.0,
            1.0 if column.dtype == "int64" else 0.0,
            1.0 if column.dtype == "object" else 0.0,
        ]

    def fit(self, dict_of_databases):
        """Fit this solver.

        Args:
            dict_of_databases (dict):
                Map from database names to tuples containing ``MetaData``
                instances and table dictionaries, which contain table names
                as input and ``pandas.DataFrames`` as values.
        """
        X, y = [], []
        iterator = tqdm(dict_of_databases.items())
        for database_name, (metadata, tables) in iterator:
            iterator.set_description("Extracting features from %s" % database_name)
            for table in metadata.get_tables():
                if "primary_key" not in table:
                    pk = []
                elif not isinstance(table["primary_key"], str):
                    pk = table["primary_key"]
                else:
                    pk = [table["primary_key"]]

                table["primary_key"]
                for column in tables[table["name"]].columns:
                    X.append(self._feature_vector(tables[table["name"]], column))
                    y.append(1.0 if column in pk else 0.0)

        X, y = np.array(X), np.array(y)

        self.model = RandomForestClassifier(*self._model_args, **self._model_kwargs)
        self.model.fit(X, y)

        if isinstance(self._threshold, list):
            best_f1 = -float('inf')
            best_threshold = None
            pred_y = self.model.predict(X)
            len_true = sum(y)
            for threshold in self._threshold:
                filtered_y = (pred_y >= threshold).astype(float)
                intersect = sum(filtered_y * y)
                len_pred = sum(filtered_y)
                if intersect * len_true * len_pred == 0:
                    f1 = 0
                else:
                    precision = intersect / len_pred
                    recall = intersect / len_true
                    f1 = 2.0 * precision * recall / (precision + recall)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            self._threshold = best_threshold

    def _score_all_keys(self, table):
        return [(column, self.model.predict([self._feature_vector(table, column)]))
                for column in table.columns]

    def _find_primary_key(self, table):
        ret = []
        for column, score in self._score_all_keys(table):
            if score >= self._threshold:
                ret.append(column)

        return ret

    def solve(self, tables):
        """Solve the problem.

        The output is a dictionary contiaining table names as keys, and the
        name of the field that is most likely to be the primary key as values.

        Args:
            tables (dict):
                Dict containing table names as input and ``pandas.DataFrames``
                as values.

        Returns:
            dict:
                Dict containing table names as keys and field names as values.
        """
        primary_keys = {}
        for table in tables:
            primary_keys[table] = self._find_primary_key(tables[table])

        return primary_keys
