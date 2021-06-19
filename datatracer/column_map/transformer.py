import numpy as np


class Transformer:

    def __init__(self, tables, foreign_keys):
        """
        The `Transformer` class provides an interface between the database and
        the column mapping solver.

        For example, during the forwards pass, it could take a date field and
        transform into three columns - day, month, and year - which will allow
        the solver to potentially identify the lineage of some target column.

        Then, during the backwards pass, it can take the scores produced by the
        solver (i.e. the solver assigns scores indicating how important the day,
        month, and year columns are to predicting the target column) and combine
        them into a single score for the `date` field.
        """
        self.tables = tables
        self.foreign_keys = foreign_keys

    def forward(self, table, field):
        """
        This function returns a (X, y) tuple containing numerical values which
        is suitable for machine learning libraries. The `X` array contains
        data from columns that are potentially related to the target field. The
        `y` array contains the values of the target field.
        """
        df = self.tables[table]
        df = df.select_dtypes("number")
        df = df.fillna(0.0)
        X, y = df.drop([field], axis=1), df[field]
        self.columns = [(table, col_name) for col_name in X.columns]
        X, y = X.values, y.values

        X_new, columns_new = self._get_counts(table)
        if columns_new:
            X = np.concatenate([X, X_new], axis=1)
            self.columns.extend(columns_new)

        X_new, columns_new = self._get_aggregations(table)
        if columns_new:
            X = np.concatenate([X, X_new], axis=1)
            self.columns.extend(columns_new)

        return X, y

    def _get_counts(self, table):
        """
        Get the foreign keys where the given table is the parent.
        """
        X, columns = [], []
        for fk in self.foreign_keys:
            if fk["ref_table"] != table:
                continue

            # Count the number of rows for each key.
            child_table = self.tables[fk["table"]].copy()
            child_table["_dummy_"] = 0.0
            child_counts = child_table.groupby(fk["field"]).count().iloc[:, 0:1]
            child_counts.columns = ["_tmp_"]

            # Merge the counts into the parent table
            parent_table = self.tables[table]
            parent_table = parent_table.set_index(fk["ref_field"])
            parent_table = parent_table.join(child_counts).reset_index()

            X.append(parent_table["_tmp_"].fillna(0.0).values)
            columns.append((fk["table"], fk["field"]))

        return np.array(X).transpose(), columns

    def _get_aggregations(self, table):
        """
        Get the foreign keys where the given table is the parent.
        """
        X, columns = [], []
        for fk in self.foreign_keys:
            if fk["ref_table"] != table:
                continue

            for op, op_name in [
                (lambda x: x.sum(), "SUM"),
                (lambda x: x.max(), "MAX"),
                (lambda x: x.min(), "MIN"),
                (lambda x: x.std(), "STD"),
            ]:
                # Count the number of rows for each key.
                child_table = self.tables[fk["table"]].copy()
                if len(child_table.columns) <= 1:
                    continue

                child_counts = op(child_table.groupby(fk["field"]))
                old_column_names = list(child_counts.columns)
                child_counts.columns = ["%s(%s)" % (op_name, col_name)
                                        for col_name in old_column_names]

                # Merge the counts into the parent table
                parent_table = self.tables[table]
                parent_table = parent_table.set_index(fk["ref_field"])
                parent_table = parent_table.join(child_counts).reset_index()

                for old_name, col_name in zip(old_column_names, child_counts.columns):
                    if parent_table[col_name].dtype.kind == "f":
                        X.append(parent_table[col_name].fillna(0.0).values)
                        columns.append((fk["table"], old_name))

        return np.array(X).transpose(), columns

    def backward(self, feature_importances):
        """
        This function takes an array of `feature_importances` which corresponds
        to the `X` matrix produced by the last call to `forward`. It returns a
        mapping from fields to importance scores.
        """
        obj = {}
        for column, importance in zip(self.columns, feature_importances):
            if column in obj:
                obj[column] += importance
            else:
                obj[column] = importance

        return obj
