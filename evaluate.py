import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from llama_cpp import Llama
from llama_cpp import LlamaGrammar

from pathlib import Path
import random
import json
import shutil
import sys

# necessary properties are absent
# returns True if check failed
def check_necessary_parameters(gt_json, pred_json):
    # not iterable => necessary parameters are absent
    try:
        iter(pred_json)
    except:
        return True
    if isinstance(gt_json, str):
        return True
    for key, val in gt_json.items():
        if key not in pred_json:
            return True
        if type(val) == dict:
            if check_necessary_parameters(val, pred_json[key]):
                return True
    return False

# some additional from method doc (with or without necessary)
def check_additional_parameters(gt_json, pred_json, json_scheme):
    for key, val in pred_json.items():
        if key not in json_scheme:
                continue
        if key not in gt_json:
            return True
        if type(val) == dict:
            # hallucinated => skipped 
            try:
                if check_additional_parameters(gt_json[key], val, json_scheme[key]['properties']):
                    return True
            except Exception as e:
                print(e)
                print(key, pred_json, gt_json, json_scheme, sep='\n')
    return False

# hallucinated properties that aren’t described in method doc
def check_hallucinated_parameters(pred_json, json_scheme):
    for key, val in pred_json.items():
        if key not in json_scheme:
            return True
        if type(val) == dict:
            # properties don't exist == sth hallucinated
            try:
                if check_hallucinated_parameters(val, json_scheme[key]['properties']):
                    return True
            except:
                return True
    return False

# incorrect value of parameter from prediction
def check_correctness_parameters(gt_json, pred_json):
    # not incorrect == hallucinated
    try:
        iter(gt_json)
    except:
        return False
    if isinstance(gt_json, str):
        return False
    for key, val in pred_json.items():
        if key not in gt_json:
                continue
        if type(val) == dict:
            if check_correctness_parameters(gt_json[key], val):
                return True
        elif gt_json[key] != val:
            return True
    return False

def check_correctness_json(gt_json, pred_json):
    return gt_json == pred_json

def compare_gt_pred(output_df, gt_df, json_schemes_df):
    merged_df = gt_df.merge(output_df, how='inner', on='id', suffixes=("_gt", "_pred"))
    compared_df = pd.DataFrame(columns=['id', 'device_name', 'env', 'user_cmd', 'gt_mtd', 'pred_mtd', 'gt_json_cmd', 'pred_json_cmd',
                                               'retriever_cor', 'json_cor', 'add', 'hall', 'absent', 'incor'])
    for _, row in merged_df.iterrows():
        methods_names_pred = row['mtd_pred'].split(',')
        try:
            gt_json = json.loads(row['json_cmd_gt'])
            pred_json = json.loads(row['json_cmd_pred'])
        except Exception as ex:
            print(ex)
            print(row['id'])
            print(row['json_cmd_gt'])
            print(row['json_cmd_pred'])
            continue
        try:
            method_name = pred_json['method']
            method_df = json_schemes_df[json_schemes_df['method'] == method_name]
            if method_df.shape[0] == 0:
                json_scheme = None
            else:
                json_scheme = json.loads(method_df.iloc[0]['json'])
        except:
            json_scheme = None

        compared_dict = {'id': row['id'], 'device_name': row['device_name'], 'env': row['env'],
            'user_cmd': row['user_cmd'], 'gt_mtd': row['mtd_gt'],
            'pred_mtd': row['mtd_pred'], 'gt_json_cmd': row['json_cmd_gt'], 'pred_json_cmd': row['json_cmd_pred']}
        
        compared_dict['retriever_cor'] = row['mtd_gt'] in methods_names_pred
        compared_dict['json_cor'] = check_correctness_json(gt_json, pred_json)
        compared_dict['absent'] = check_necessary_parameters(gt_json, pred_json)
        compared_dict['incor'] = check_correctness_parameters(gt_json, pred_json)
        if json_scheme:
            compared_dict['add'] = check_additional_parameters(gt_json, pred_json, json_scheme)
            compared_dict['hall'] = check_hallucinated_parameters(gt_json, json_scheme)
        else:
            compared_dict['add'] = False
            compared_dict['hall'] = False

        compared_df.loc[len(compared_df)] = pd.Series(compared_dict)
    return compared_df

def evaluate(gt_df, output_df, json_schemes_df, run_name, settings, output_dir, save_intermediate=False, verbose=False):
    compared_df = compare_gt_pred(output_df, gt_df, json_schemes_df)
    if save_intermediate:
        compared_df.to_csv(output_dir / run_name / 'compared.csv', index=False)
    if verbose:
        print(len(gt_df))
        print(len(compared_df))

    results_df = pd.DataFrame(columns=['run_name', 'settings', 'method_acc', 'json_acc'])

    method_acc = compared_df[compared_df['retriever_cor'] == True].shape[0] / compared_df.shape[0]
    json_acc = compared_df[compared_df['json_cor'] == True].shape[0] / compared_df.shape[0]
    results_df.loc[0] = pd.Series({'run_name': run_name,
                                    'settings': json.dumps(settings),
                                    'method_acc': method_acc,
                                    'json_acc': json_acc})
    results_df.to_csv(output_dir / 'results.csv', index=False, header=not (output_dir / 'results.csv').exists(), mode='a')

    fail_reasons_df = compared_df[compared_df['retriever_cor'] == True].copy(deep=True)
    fail_reasons_df = fail_reasons_df.loc[:, ['gt_mtd', 'retriever_cor', 'json_cor', 'add', 'hall', 'absent', 'incor']]
    fail_reasons_df.rename(columns={'gt_mtd': 'method'}, inplace=True)
    fail_reasons_df = fail_reasons_df.replace({True: 1, False: 0})
    fail_reasons_df = fail_reasons_df.groupby('method', as_index=False).sum()
    fail_reasons_df.insert(0, 'run_name', run_name)
    fail_reasons_df.to_csv(output_dir / 'fail_reasons.csv', index=False, header=not (output_dir / 'fail_reasons.csv').exists(), mode='a')

if __name__ == '__main__':
    gt = json.loads("""{"method": "Input.ResetCounters", "params": {"id": 53, "type": ["total"]}}""")
    pred = json.loads("""{"method": "Input.ResetCounters", "params": {"id": 53}}""")
    scheme = json.loads("""{"method": "Input.ResetCounters", "params": {"type": "object", "properties": {"id": {"type": "number", "description": "Id of the Input component instance. Required"}, "type": {"type": "array of strings", "description": "Array of strings, selects which counter to reset Optional"}}}}""")
    print(check_additional_parameters(gt, pred, scheme))