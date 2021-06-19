from collections import Counter
from itertools import permutations

from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from datatracer.foreign_key.base import ForeignKeySolver


class StandardForeignKeySolver(ForeignKeySolver):

    def __init__(self, threshold=[i / 20 for i in range(20)], add_details=False, *args, **kwargs):
        self._threshold = threshold
        self._add_details = add_details
        self._model_args = args
        self._model_kwargs = kwargs

    def _diff(self, a, b):
        diff = 0
        a, b = Counter(a), Counter(b)
        for k in set(a.keys()).union(set(b.keys())):
            diff += abs(a[k] - b[k])

        return diff

    def _feature_vector(self, parent_col, child_col):
        parent_set, child_set = set(parent_col.unique()), set(child_col.unique())
        len_intersect = len(parent_set.intersection(child_set))
        return [
            len_intersect / (len(child_set) + 1e-5),
            len_intersect / (len(parent_set) + 1e-5),
            len_intersect / (max(len(child_set), len(parent_set)) + 1e-5),
            1.0 if parent_col.name == child_col.name else 0.0,
            self._diff(parent_col.name, child_col.name),
            1.0 if child_set.issubset(parent_set) else 0.0,
            1.0 if parent_set.issubset(child_set) else 0.0,
            len(parent_set) / (len(parent_col) + 1e-5),
            1.0 if "key" in parent_col.name else 0.0,
            1.0 if "id" in parent_col.name else 0.0,
            1.0 if "_key" in parent_col.name else 0.0,
            1.0 if "_id" in parent_col.name else 0.0,
            1.0 if parent_col.dtype == "int64" else 0.0,
            1.0 if parent_col.dtype == "object" else 0.0,
            len(child_set) / (len(child_col) + 1e-5),
            1.0 if "key" in child_col.name else 0.0,
            1.0 if "id" in child_col.name else 0.0,
            1.0 if "_key" in child_col.name else 0.0,
            1.0 if "_id" in child_col.name else 0.0,
            1.0 if child_col.dtype == "int64" else 0.0,
            1.0 if child_col.dtype == "object" else 0.0,
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
            fks = metadata.get_foreign_keys()
            tables_info = {table_info['name']: table_info for table_info in metadata.get_tables()}
            fks_new = []
            for fk in fks:
                if isinstance(fk["field"], list):
                    for field, ref_field in zip(fk["field"], fk["ref_field"]):
                        fks_new.append((fk["table"], field, fk["ref_table"], ref_field))
                else:
                    fks_new.append((fk["table"], fk["field"], fk["ref_table"], fk["ref_field"]))
            fks = set(fks_new)

            for t1, t2 in permutations(tables.keys(), r=2):
                table = tables_info[t2]
                if "primary_key" not in table:
                    t2_columns = tables[t2].columns
                elif not isinstance(table["primary_key"], str):
                    t2_columns = table["primary_key"]
                else:
                    t2_columns = [table["primary_key"]]

                for c2 in t2_columns:
                    for c1 in tables[t1].columns:
                        if tables[t1][c1].dtype.kind != tables[t2][c2].dtype.kind:
                            continue

                        if len(tables[t1][c1]) == 0 or len(tables[t2][c2]) == 0:
                            print(tables.keys())
                            raise RuntimeError("Found empty table!")

                        X.append(self._feature_vector(tables[t2][c2], tables[t1][c1]))
                        y.append(1.0 if (t1, c1, t2, c2) in fks else 0.0)

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

    def solve(self, tables, primary_keys=None):
        """Solve the foreign key detection problem.

        The output is a list of foreign key specifications, in order from the most likely
        to the least likely.

        Args:
            tables (dict):
                Dict containing table names as input and ``pandas.DataFrames``
                as values.
            primary_keys (dict):
                (Ignored). This particular implementation does not use this argument.

        Returns:
            dict:
                List of foreign key specifications, sorted by likelyhood.
        """
        X, foreign_keys = [], []
        for t1, t2 in permutations(tables.keys(), r=2):
            for c1 in tables[t1].columns:
                if (primary_keys is None) or (t2 not in primary_keys):
                    t2_columns = tables[t2].columns
                elif not isinstance(primary_keys[t2], str):
                    t2_columns = primary_keys[t2]
                else:
                    t2_columns = [primary_keys[t2]]
                for c2 in t2_columns:
                    if tables[t1][c1].dtype.kind != tables[t2][c2].dtype.kind:
                        continue

                    X.append(self._feature_vector(tables[t2][c2], tables[t1][c1]))
                    foreign_keys.append((t1, c1, t2, c2))

        if not foreign_keys:
            return []

        scores = self.model.predict(X)

        best_foreign_keys = []
        for score, (t1, c1, t2, c2) in sorted(zip(scores, foreign_keys), reverse=True):
            if self._threshold is None or score >= self._threshold:
                foreign_key = {
                    "table": t1,
                    "field": c1,
                    "ref_table": t2,
                    "ref_field": c2,
                }
                if self._add_details:
                    foreign_key.update({
                        "score": score,
                        "features": self._feature_vector(tables[t2][c2], tables[t1][c1])
                    })

                best_foreign_keys.append(foreign_key)

        return best_foreign_keys
