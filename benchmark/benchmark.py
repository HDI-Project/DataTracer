import argparse
import os
import queue
import threading
import time
from io import BytesIO
from time import ctime, time
from urllib.parse import urljoin
from urllib.request import urlopen
from zipfile import ZipFile

import boto3
import dask
import pandas as pd
from dask.diagnostics import ProgressBar

from datatracer import DataTracer, load_datasets, sample_datasets

BUCKET_NAME = 'tracer-data'
DATA_URL = 'http://{}.s3.amazonaws.com/'.format(BUCKET_NAME)


def download(data_dir):
    """Download benchmark datasets from S3.

    This downloads the benchmark datasets from S3 into the target folder in an 
    uncompressed format. It skips datasets that have already been downloaded.

    Please make sure an appropriate S3 credential is installed before you call
    this method.

    Args:
        data_dir: The directory to download the datasets to.

    Returns:
        A DataFrame describing the downloaded datasets.

    Raises:
        NoCredentialsError: If AWS S3 credentials are not found.
    """
    rows = []
    client = boto3.client('s3')
    for dataset in client.list_objects(Bucket=BUCKET_NAME)['Contents']:
        if not '.zip' in dataset['Key']:
            continue
        rows.append(dataset)
        dataset_name = dataset['Key'].replace(".zip", "")
        dataset_path = os.path.join(data_dir, dataset_name)
        if os.path.exists(dataset_path):
            dataset["Status"] = "Skipped"
            print("Skipping %s" % dataset_name)
        else:
            dataset["Status"] = "Downloaded"
            print("Downloading %s" % dataset_name)
            with urlopen(urljoin(DATA_URL, dataset['Key'])) as fp:
                with ZipFile(BytesIO(fp.read())) as zipfile:
                    zipfile.extractall(dataset_path)
    return pd.DataFrame(rows)


@dask.delayed
def primary_key(solver, target, datasets):
    """Benchmark the primary key solver on the target dataset.

    Args:
        solver: The name of the primary key pipeline.
        target: The name of the target dataset.
        datases: A dictionary mapping dataset names to (metadata, tables) tuples.

    Returns:
        A dictionary mapping metric names to values.
    """
    datasets = datasets.copy()
    metadata, tables = datasets.pop(target)

    tracer = DataTracer(solver)
    tracer.fit(datasets)

    y_true = {}
    for table in metadata.get_tables():
        if "primary_key" not in table:
            y_true[table["name"]] = set()
        elif not isinstance(table["primary_key"], str):
            y_true[table["name"]] = set(table["primary_key"])
        else:
            y_true[table["name"]] = set([table["primary_key"]])

    """
    if len(y_true) == 0:
        return {}  # Skip dataset, no primary keys found.
    """

    correct, total_pred, total_true = 0, 0, 0

    try:
        start = time()
        y_pred = tracer.solve(tables)
        end = time()
    except:
        return {
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "inference_time": 0,
            "status": "ERROR"
        }
    for table_name, primary_key in y_true.items():
        ans = y_pred.get(table_name)
        if isinstance(ans, str):
            ans = set([ans])
        else:
            ans = set(ans)
        correct += len(ans.intersection(primary_key))
        total_pred += len(ans)
        total_true += len(primary_key)

    if correct == 0 or total_pred == 0 or \
            total_true == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "inference_time": end - start,
            "status": "OK"
        }
    precision = correct / total_pred
    recall = correct / total_true
    f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "inference_time": end - start,
        "status": "OK"
    }


def benchmark_primary_key(data_dir, dataset_name=None, solver="datatracer.primary_key.basic"):
    """Benchmark the primary key solver.

    This uses leave-one-out validation and evaluates the performance of the 
    solver on the specified datasets.

    Args:
        data_dir: The directory containing the datasets.
        dataset_name: The target dataset to test on. If none is provided, will test on all available datasets by default.
        solver: The name of the primary key pipeline.

    Returns:
        A DataFrame containing the benchmark resuls.
    """
    datasets = load_datasets(data_dir)
    dataset_names = list(datasets.keys())
    if dataset_name is not None:
        if dataset_name in dataset_names:
            dataset_names = [dataset_name]
        else:
            return None
    datasets = dask.delayed(datasets)
    dataset_to_metrics = {}
    for dataset_name in dataset_names:
        dataset_to_metrics[dataset_name] = primary_key(
            solver=solver, target=dataset_name, datasets=datasets)

    with ProgressBar():
        results = dask.compute(dataset_to_metrics)[0]
    for dataset_name, metrics in results.items():
        metrics["dataset"] = dataset_name
    df = pd.DataFrame(list(results.values()))
    dataset_col = df.pop('dataset')
    df.insert(0, 'dataset', dataset_col)
    return df


@dask.delayed
def foreign_key(solver, target, datasets):
    """Benchmark the foreign key solver on the target dataset.

    Args:
        solver: The name of the foreign key pipeline.
        target: The name of the target dataset.
        datasets: A dictionary mapping dataset names to (metadata, tables) tuples.

    Returns:
        A dictionary mapping metric names to values.
    """
    datasets = datasets.copy()
    metadata, tables = datasets.pop(target)

    tracer = DataTracer(solver)
    tracer.fit(datasets)

    y_true = set()
    for fk in metadata.get_foreign_keys():
        if not isinstance(fk["field"], str):
            continue  # Skip composite foreign keys
        y_true.add((fk["table"], fk["field"], fk["ref_table"], fk["ref_field"]))

    try:
        start = time()
        fk_pred = tracer.solve(tables)
        end = time()
    except:
        return {
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "inference_time": 0,
            "status": "ERROR"
        }

    y_pred = set()
    for fk in fk_pred:
        y_pred.add((fk["table"], fk["field"], fk["ref_table"], fk["ref_field"]))

    if len(y_pred) == 0 or len(y_true) == 0 or \
            len(y_true.intersection(y_pred)) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "inference_time": end - start,
            "status": "OK"
        }

    precision = len(y_true.intersection(y_pred)) / len(y_pred)
    recall = len(y_true.intersection(y_pred)) / len(y_true)
    f1 = 2.0 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "inference_time": end - start,
        "status": "OK"
    }


def benchmark_foreign_key(data_dir, dataset_name=None, solver="datatracer.foreign_key.standard"):
    """Benchmark the foreign key solver.

    This uses leave-one-out validation and evaluates the performance of the 
    solver on the specified datasets.

    Args:
        data_dir: The directory containing the datasets.
        dataset_name: The target dataset to test on. If none is provided, will test on all available datasets by default.
        solver: The name of the foreign key pipeline.

    Returns:
        A DataFrame containing the benchmark resuls.
    """
    datasets = load_datasets(data_dir)
    #datasets = sample_datasets(datasets, max_size=20)
    dataset_names = list(datasets.keys())
    if dataset_name is not None:
        if dataset_name in dataset_names:
            dataset_names = [dataset_name]
        else:
            return None
    datasets = dask.delayed(datasets)
    dataset_to_metrics = {}
    for dataset_name in dataset_names:
        dataset_to_metrics[dataset_name] = foreign_key(
            solver=solver, target=dataset_name, datasets=datasets)

    with ProgressBar():
        results = dask.compute(dataset_to_metrics)[0]
    for dataset_name, metrics in results.items():
        metrics["dataset"] = dataset_name
    df = pd.DataFrame(list(results.values()))
    dataset_col = df.pop('dataset')
    df.insert(0, 'dataset', dataset_col)
    return df


@dask.delayed
def evaluate_single_column_map(constraint, tracer, tables):
    field = constraint["fields_under_consideration"][0]
    related_fields = constraint["related_fields"]

    y_true = set()
    for related_field in related_fields:
        y_true.add((related_field["table"], related_field["field"]))

    try:
        start = time()
        y_pred = tracer.solve(tables, target_table=field["table"], target_field=field["field"])
        y_pred = {field for field, score in y_pred.items() if score > 0.0}
        end = time()
    except:
        return {
            "table": field["table"],
            "field": field["field"],
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "inference_time": 0,
            "status": "ERROR"
        }

    if len(y_pred) == 0 or len(y_true) == 0 or \
            len(y_true.intersection(y_pred)) == 0:
        return {
            "table": field["table"],
            "field": field["field"],
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "inference_time": end - start,
            "status": "OK"
        }
    else:
        precision = len(y_true.intersection(y_pred)) / len(y_pred)
        recall = len(y_true.intersection(y_pred)) / len(y_true)
        f1 = 2.0 * precision * recall / (precision + recall)

        return {
            "table": field["table"],
            "field": field["field"],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "inference_time": end - start,
            "status": "OK"
        }

@dask.delayed
def column_map(solver, target, datasets):
    """Benchmark the column map solver on the target dataset.

    Args:
        solver: The name of the column map pipeline.
        target: The name of the target dataset.
        datases: A dictionary mapping dataset names to (metadata, tables) tuples.

    Returns:
        A list of dictionaries mapping metric names to values for each deived column.
    """
    datasets = datasets.copy()
    metadata, tables = datasets.pop(target)
    if not metadata.data.get("constraints"):
        return {}  # Skip dataset, no constraints found.

    tracer = DataTracer(solver)
    tracer.fit(datasets)

    list_of_metrics = []
    for constraint in metadata.data["constraints"]:
        list_of_metrics.append(evaluate_single_column_map(constraint, tracer, tables))

    list_of_metrics = dask.compute(list_of_metrics)[0]
    return list_of_metrics


def benchmark_column_map(data_dir, dataset_name=None, solver="datatracer.column_map.basic"):
    """Benchmark the column map solver.

    This uses leave-one-out validation and evaluates the performance of the 
    solver on the specified datasets.

    Args:
        data_dir: The directory containing the datasets.
        dataset_name: The target dataset to test on. If none is provided, will test on all available datasets by default.
        solver: The name of the column map pipeline.

    Returns:
        A DataFrame containing the benchmark resuls.
    """
    datasets = load_datasets(data_dir)
    #datasets = sample_datasets(datasets, max_size=20)
    dataset_names = list(datasets.keys())
    if dataset_name is not None:
        if dataset_name in dataset_names:
            dataset_names = [dataset_name]
        else:
            return None
    datasets = dask.delayed(datasets)
    dataset_to_metrics = {}
    for dataset_name in dataset_names:
        dataset_to_metrics[dataset_name] = column_map(
            solver=solver, target=dataset_name, datasets=datasets)

    rows = []
    with ProgressBar():
        results = dask.compute(dataset_to_metrics)[0]
    for dataset_name, list_of_metrics in results.items():
        for metrics in list_of_metrics:
            metrics["dataset"] = dataset_name
            rows.append(metrics)
    df = pd.DataFrame(rows)
    dataset_col = df.pop('dataset')
    table_col = df.pop('table')
    field_col = df.pop('field')
    df.insert(0, 'field', field_col)
    df.insert(0, 'table', table_col)
    df.insert(0, 'dataset', dataset_col)
    return df


def start_with(target, source):
    return len(source) <= len(target) and target[:len(source)] == source


def aggregate(cmd_name):
    cmd_abbrv = { 'column': 'ColMap_st',
    'foreign': 'ForeignKey_st',
    'primary': 'PrimaryKey_st'
    }
    if cmd_name not in cmd_abbrv:
        print("Invalid command name!")
        return None #invalid command name
    cmd_name = cmd_abbrv[cmd_name]
    dfs = []
    for file in os.listdir("Reports"):
        if start_with(file, cmd_name):
            dfs.append(pd.read_csv("Reports/"+file))
    if len(dfs) == 0:
        print("No available test results!")
        return None
    df = pd.concat(dfs, axis=0, ignore_index=True)
    os.system("rm Reports/"+cmd_name+"*") #Clean up the caches
    return df


def _get_parser():
    shared_args = argparse.ArgumentParser(add_help=False)
    shared_args.add_argument('--data_dir', type=str, 
        default=os.path.expanduser("~/tracer_data"), required=False, 
        help='Path to the benchmark datasets.')
    default_csv = "report_" + ctime().replace(" ", "_") + ".csv"
    default_csv = default_csv.replace(":", "_")
    shared_args.add_argument('--csv', type=str,
        default=os.path.expanduser(default_csv), required=False, 
        help='Path to the CSV file where the report will be written.')
    shared_args.add_argument('--ds_name', type=str,
        default=None, required=False, 
        help='Name of the dataset to test on. Default is all available datasets.')
    shared_args.add_argument('--problem', type=str,
        default=None, required=False, 
        help='Name of the tests results to aggregate.')
    shared_args.add_argument('--primitive', type=str,
        default=None, required=False, 
        help='Name of the primitive to be tested.')

    parser = argparse.ArgumentParser(
        prog='datatracer-benchmark',
        description='DataTracer Benchmark CLI'
    )

    command = parser.add_subparsers(title='command', help='Command to execute')
    parser.set_defaults(benchmark=None)

    subparser = command.add_parser(
        'download',
        parents=[shared_args],
        help='Download datasets from S3.'
    )
    subparser.set_defaults(command=download)

    subparser = command.add_parser(
        'primary',
        parents=[shared_args],
        help='Primary key benchmark.'
    )
    subparser.set_defaults(command=benchmark_primary_key)

    subparser = command.add_parser(
        'foreign',
        parents=[shared_args],
        help='Foreign key benchmark.'
    )
    subparser.set_defaults(command=benchmark_foreign_key)

    subparser = command.add_parser(
        'column',
        parents=[shared_args],
        help='Column map benchmark.'
    )
    subparser.set_defaults(command=benchmark_column_map)

    subparser = command.add_parser(
        'aggregate',
        parents=[shared_args],
        help='Aggregate separate test results.'
    )
    subparser.set_defaults(command=aggregate)

    return parser


def main():
    parser = _get_parser()
    args = parser.parse_args()
    if args.command == download:
        df = args.command(args.data_dir)
    elif args.command == aggregate:
        df = args.command(args.problem)
    else:
        if args.primitive is None:
            df = args.command(args.data_dir, args.ds_name)
        else:
            df = args.command(args.data_dir, args.ds_name, solver=args.primitive)
    cmd_abbrv = { 'column': 'ColMap_',
    'foreign': 'ForeignKey_',
    'primary': 'PrimaryKey_'
    }
    cmd_str = { benchmark_column_map: 'ColMap_',
    benchmark_foreign_key: 'ForeignKey_',
    benchmark_primary_key: 'PrimaryKey_',
    aggregate: cmd_abbrv[args.problem] if args.problem in cmd_abbrv else ''
    }
    csv_name = "st_" + args.ds_name + ".csv" if args.ds_name else args.csv
    # st is for recognition in the aggregation step

    if csv_name and (args.command in cmd_str) and (df is not None):
        df.to_csv("Reports/" + cmd_str[args.command] + csv_name, index=False)
    print(df)


if __name__ == "__main__":
    main()
