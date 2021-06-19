# -*- coding: utf-8 -*-

"""DataTracer core module.

This module introduces tools for sampling from databases while respecting the row lineage.
"""

import random

import dask
from dask.diagnostics import ProgressBar


def calculate_size(transformed_dataset):
    """Helper function to calculate the total size of a dataset

    Args:
        transformed_dataset (dict): a ``TransformedDataset`` instance, which maps (str) table name
            to {'size': (float) size of the table in byte, 'row_size': (float) the size of a
            row in byte, 'entries': (set) the column names, 'chosen': (set) the rows selected}

    Returns:
        float: the dataset size in byte
    """
    size = 0
    for table in transformed_dataset.values():
        size += table['row_size'] * len(table['chosen'])
    return size


def transform_dataset(metadata, dataset):
    """Pack the foreign key relations, sizes, and the rows selected of a dataset into dictionaries.

    Args:
        metadata (dict): a ``MetaData`` instance
        dataset (dict): maps table name to pd.DataFrame object

    Returns:
        dict: a ``TransformedForeignKey`` instance, which maps (str) table name to (list(tuple))
            its associated foreign key relations
        dict: a ``TransformedDataset`` instance
        float: the dataset size in byte
    """
    fks = metadata.get_foreign_keys()
    transformed_fk = {}
    key_columns = {table_name: set() for table_name in dataset}
    for fk in fks:
        table, all_field, ref_table, all_ref_field =\
            fk["table"], fk["field"], fk["ref_table"], fk["ref_field"]
        if isinstance(all_field, str):
            all_field = [all_field]
            all_ref_field = [all_ref_field]
        for field, ref_field in zip(all_field, all_ref_field):
            key_columns[table].add(field)
            key_columns[ref_table].add(ref_field)
            if ref_table not in transformed_fk:
                transformed_fk[ref_table] = []
            transformed_fk[ref_table].append((ref_table, ref_field, table, field))
    transformed_dataset = {}
    size = 0
    for table_name in dataset:
        table = dataset[table_name]
        columns = key_columns[table_name]
        transformed_table = {'size': table.memory_usage().sum(),
                             'row_size': float(table.memory_usage().sum()) / len(table),
                             'entries': {col: {} for col in columns},
                             'chosen': set(range(len(table))), }
        for idx in range(len(table)):
            for col in columns:
                val = table.iloc[idx][col]
                if val not in transformed_table['entries'][col]:
                    transformed_table['entries'][col][val] = []
                transformed_table['entries'][col][val].append(idx)
        transformed_dataset[table_name] = transformed_table
        size += transformed_table['size']
    return transformed_fk, transformed_dataset, size


def backward_transform(transformed_dataset, dataset):
    """Transform a ``TransformedDataset`` instance back into a dictionary mapping name
        to pd.DataFrame objects

    Args:
        transformed_dataset (dict): a ``TransformedDataset`` instance
        dataset (dict): a dictionary mapping table name to pd.DataFrame object

    Returns:
        dict: a dictionary mapping table name to pd.DataFrame object
    """
    new_dataset = {}
    for table_name in dataset:
        idxes = list(transformed_dataset[table_name]['chosen'])
        new_dataset[table_name] = dataset[table_name].iloc[idxes]
    return new_dataset


def remove_row(dataset, transformed_fk, transformed_dataset, table_name, idx):
    """Remove a row from a table, and recursively removed all other rows associated
        by some foreign key relations

    Args:
        dataset (dict): a dictionary mapping table name to pd.DataFrame object
        transformed_fk (dict): a ``TransformedForeignKey`` instance
        transformed_dataset (dict): a ``TransformedDataset`` instance
        table_name (string): the name of the table in which a row is to be removed
        idx (int): the index of the row to be removed

    Returns:
        None if at least of the tables is empty after the recursive removal. True otherwise.
    """
    if idx in transformed_dataset[table_name]['chosen']:
        transformed_dataset[table_name]['chosen'].remove(idx)
        if len(transformed_dataset[table_name]['chosen']) == 0:
            return None
    row = dataset[table_name].iloc[idx]
    if table_name in transformed_fk:
        for table, col, other_table, other_col in transformed_fk[table_name]:
            val = row[col]
            if val in transformed_dataset[other_table]['entries'][other_col]:
                for new_idx in transformed_dataset[other_table]['entries'][other_col][val]:
                    if new_idx in transformed_dataset[other_table]['chosen']:
                        if remove_row(dataset, transformed_fk, transformed_dataset,
                                      other_table, new_idx) is None:
                            return None
    return True


def get_root_tables(metadata):
    """Get all root tables (tables who are never a child table in a foreign key relation)
        of the dataset.

    Args:
        metadata (dict): a ``MetaData`` instance

    Returns:
        set: the root table names
    """
    all_tables = {table['name'] for table in metadata.get_tables()}

    for fk in metadata.get_foreign_keys():
        if fk['table'] in all_tables:
            all_tables.remove(fk['table'])
    if len(all_tables) > 0:
        return all_tables
    else:
        return {table['name'] for table in metadata.get_tables()}


@dask.delayed
def sample_dataset(metadata=None, dataset=None, max_size=None, max_ratio=1.0, drop_threshold=None,
                   rand_seed=0, database_name=None, dict_of_databases=None):
    """Sample from a dataset

    Args:
        metadata (dict): a ``MetaData`` instance.
            Will be extracted from datasets if None is provided.
        dataset (dict): maps table name to pd.DataFrame object.
            Will be extracted from datasets if None is provided.
        max_size (float): the target size to sample down to, in MB.
        max_ratio (float): the target fraction to sample down to.
        drop_threshold (float): the maximum dataset size allowed.
        rand_seed (int): seed for random.
        database_name (str): name of the dataset.
        dict_of_databases (dict): maps (str) dataset name to (metadata, dataset) tuple.

    Returns:
        tuple (None, str) describing the reason of dropping the dataset if it is dropped.
        dict, which maps table name to pd.DataFrame object, otherwise.
    """
    if dict_of_databases is not None:
        metadata, dataset = dict_of_databases[database_name]

    if len(dataset) == 0:
        return (None, 'empty')  # empty dataset

    size = sum([table.memory_usage().sum() for _, table in dataset.items()])
    if max_size is not None:
        max_size *= (1024.0**2)  # input max_size is in MB
    else:
        max_size = size
    target_size = min(max_size, size * float(max_ratio))

    if drop_threshold is None:
        drop_threshold = 5 * target_size
    if size <= target_size + 1:
        return dataset
    elif size > drop_threshold:
        return (None, 'size')

    random.seed(rand_seed)
    transformed_fk, transformed_dataset, size = transform_dataset(metadata, dataset)
    root_tables = get_root_tables(metadata)
    while calculate_size(transformed_dataset) > target_size + \
            1:  # +1 is for preventing precision issues
        table_name = random.sample(root_tables, 1)[0]
        if len(transformed_dataset[table_name]['chosen']) > 0:
            idx = random.sample(transformed_dataset[table_name]['chosen'], 1)[0]
            if remove_row(dataset, transformed_fk, transformed_dataset, table_name, idx) is None:
                return (None, 'empty')
        else:
            return (None, 'empty')

    return backward_transform(transformed_dataset, dataset)


def sample_datasets(dict_of_databases, max_size=None, max_ratio=1.0,
                    drop_threshold=None, rand_seed=0):
    """Sample from multiple datasets

    Args:
        dict_of_databases (dict): maps (str) dataset name to (metadata, dataset) tuple.
        max_size (float): the target size to sample down to, in MB.
        max_ratio (float): the target fraction to sample down to.
        drop_threshold (float): the maximum dataset size allowed.
        rand_seed (int): seed for random.

    Returns:
        dict: maps (str) dataset name to (metadata, dataset) tuple.
    """
    db_names = list(dict_of_databases.keys())
    immediate_dict_of_db = dict_of_databases
    dict_of_databases = dask.delayed(dict_of_databases)
    new_dict_of_databases = {}
    for database_name in db_names:
        new_dict_of_databases[database_name] = sample_dataset(
            max_size=max_size,
            max_ratio=max_ratio,
            drop_threshold=drop_threshold,
            rand_seed=rand_seed,
            dict_of_databases=dict_of_databases,
            database_name=database_name)
    with ProgressBar():
        new_dict_of_databases = dask.compute(new_dict_of_databases)[0]
    for database_name in db_names:
        if isinstance(new_dict_of_databases[database_name], tuple):
            if new_dict_of_databases[database_name][1] == 'empty':
                print("%s is dropped because of empty tables when sampling" % database_name)
            elif new_dict_of_databases[database_name][1] == 'size':
                print("%s is dropped because it's too big" % database_name)
            del new_dict_of_databases[database_name]
        else:
            metadata = immediate_dict_of_db[database_name][0]
            new_dict_of_databases[database_name] = (metadata, new_dict_of_databases[database_name])
    return new_dict_of_databases
