import os
import yaml
import tempfile
import zipfile
import shutil
import time
import random
import csv
import json
import torch
from minichat.core_eval import evaluate_task
from minichat.common import get_base_dir, download_file_with_lock, place_eval_bundle


EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"

def place_eval_bundle(file_path):
    # here file_path is the path to the eval_bundle.zip file
    # we need to unzip it and place it in the base directory
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        shutil.move(extracted_bundle_dir, eval_bundle_dir)
    print(f"Placed eval_bundle directory at {eval_bundle_dir}")

def evaluate_model(model, tokenizer, device, max_per_task=-1):
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    if not os.path.exists(eval_bundle_dir):
        download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", place_eval_bundle)

    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    tasks = config['icl_tasks']

    random_baselines = {}
    with open(eval_meta_data, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_name = row['Eval Task']
            random_baseline = row['Random baseline']
            random_baselines[task_name] = float(random_baseline)

    results = {}
    centered_results = {}

    for task in tasks:
        if task['icl_task_type'] != 'language_modeling':
            continue
        start_time = time.time()
        label = task['label']
        task_meta = {
            'task_type': task['icl_task_type'],
            'dataset_uri': task['dataset_uri'],
            'num_fewshot': task['num_fewshot'][0],
            'continuation_delimiter': task.get('continuation_delimiter', ' ')
        }
        data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f.readlines()]
        
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        data = data[:max_per_task] if max_per_task > 0 else data

        accuracy = evaluate_task(data, task_meta, model, tokenizer, device)
        results[label] = accuracy
        random_baseline = random_baselines.get(label, 0.0)
        centered_accuracy = (accuracy - 0.01 * random_baseline)/(1.0 - 0.01 * random_baseline)
        centered_results[label] = centered_accuracy
        end_time = time.time()
        print(f"Evaluated task {label} in {end_time - start_time:.2f} seconds: accuracy={accuracy:.4f}, centered_accuracy={centered_accuracy:.4f}")
    
    core_metric = sum(centered_results.values()) / len(centered_results)
    results['core_metric'] = core_metric
    print(f"Core Metric: {core_metric:.4f}")
    return results

            