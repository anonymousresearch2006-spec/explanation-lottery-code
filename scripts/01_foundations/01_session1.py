"""
================================================================================
THE EXPLANATION LOTTERY - SESSION 1 (FINAL Q2 GRADE)
================================================================================
Research Question: When ML models agree on predictions, do their SHAP 
explanations agree?

Datasets: 1-10 of 20 | Time: ~15 hours | Auto-saves after each dataset
Safe to interrupt - will resume from last checkpoint

Version: 1.0 (Final Corrected)
================================================================================
"""

# ============== STEP 1: INSTALL DEPENDENCIES ==============
import subprocess
import sys

print("="*70)
print("INSTALLING DEPENDENCIES...")
print("="*70)

packages = [
    "numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", 
    "catboost", "shap", "openml", "scipy", "tqdm", "matplotlib", "seaborn"
]

for package in packages:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
        print(f"  Installed: {package}")
    except:
        print(f"  Warning: Could not install {package}")

print("Dependencies ready!")
print("="*70)

# ============== STEP 2: IMPORTS ==============
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import shap
shap.initjs = lambda: None  # Suppress JS output

import openml
openml.config.set_root_cache_directory('data/openml_cache')

from scipy.stats import spearmanr
import time
from datetime import datetime
import json
import logging
import traceback

# ============== STEP 2b: UTILS IMPORTS ==============
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from scripts.utils.logging_setup import setup_logging
from scripts.utils.checkpointing import load_checkpoint, save_checkpoint
from scripts.utils.data_loading import load_and_preprocess
from scripts.utils.model_training import train_models, evaluate_models
from scripts.utils.shap_computation import get_shap_values, find_agreement_instances, compute_agreement_metrics

# ============== STEP 3: SETUP PROJECT STRUCTURE ==============
PROJECT_DIR = "explanation_lottery"
RESULTS_DIR = f"{PROJECT_DIR}/results/session1"
CHECKPOINTS_DIR = f"{PROJECT_DIR}/checkpoints"
LOGS_DIR = f"{PROJECT_DIR}/logs"
DATA_DIR = f"{PROJECT_DIR}/data"

for directory in [PROJECT_DIR, RESULTS_DIR, CHECKPOINTS_DIR, LOGS_DIR, DATA_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============== STEP 4: SETUP LOGGING ==============
logger, log_filename = setup_logging(LOGS_DIR, 'session1')

logger.info("="*70)
logger.info("THE EXPLANATION LOTTERY - SESSION 1")
logger.info("="*70)
logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"Log file: {log_filename}")
logger.info("="*70)

# ============== STEP 5: CONFIGURATION ==============
CONFIG = {
    'random_seeds': [42, 123, 456],
    'test_size': 0.2,
    'max_instances_per_dataset': 200,
    'min_agreement_instances': 10,
    'top_k_values': [3, 5, 10],
    'shap_background_samples': 100,
    'model_params': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1
    }
}

# Session 1: First 10 datasets
DATASET_IDS = [
    31,    # credit-g
    37,    # diabetes
    44,    # spambase
    50,    # tic-tac-toe
    1046,  # mozilla4
    1049,  # pc4
    1050,  # pc3
    1462,  # banknote-authentication
    1464,  # blood-transfusion
    1479,  # hill-valley
]

MODEL_NAMES = ['xgboost', 'lightgbm', 'catboost', 'random_forest', 'logistic_regression']

# Save configuration
config_file = f"{RESULTS_DIR}/experiment_config.json"
with open(config_file, 'w') as f:
    json.dump({
        'config': CONFIG,
        'dataset_ids': DATASET_IDS,
        'model_names': MODEL_NAMES,
        'session': 1,
        'started_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'version': '1.0_final'
    }, f, indent=2)

logger.info(f"Configuration:")
logger.info(f"  Datasets: {len(DATASET_IDS)}")
logger.info(f"  Models: {MODEL_NAMES}")
logger.info(f"  Seeds: {CONFIG['random_seeds']}")
logger.info(f"  Config saved: {config_file}")

# ============== STEP 6: HELPER FUNCTIONS (imported from scripts.utils) ==============
# load_and_preprocess, train_models, evaluate_models, get_shap_values,
# find_agreement_instances, compute_agreement_metrics are imported above.


def estimate_time_remaining(elapsed_seconds, completed, total):
    """Estimate remaining time based on progress."""
    if completed == 0:
        return "Calculating..."
    
    avg_time = elapsed_seconds / completed
    remaining = avg_time * (total - completed)
    
    hours = int(remaining // 3600)
    minutes = int((remaining % 3600) // 60)
    
    return f"{hours}h {minutes}m"


def validate_results(results_df):
    """Validate results dataframe before saving."""
    issues = []
    
    if len(results_df) == 0:
        issues.append("No results collected")
    
    if results_df['spearman'].isna().any():
        n_na = results_df['spearman'].isna().sum()
        issues.append(f"{n_na} NaN Spearman values")
    
    if (results_df['spearman'] < -1).any() or (results_df['spearman'] > 1).any():
        issues.append("Spearman values out of range [-1, 1]")
    
    expected_pairs = len(MODEL_NAMES) * (len(MODEL_NAMES) - 1) // 2
    pairs_found = results_df.groupby(['model_a', 'model_b']).ngroups
    if pairs_found < expected_pairs:
        issues.append(f"Only {pairs_found}/{expected_pairs} model pairs found")
    
    return issues


# ============== STEP 7: MAIN EXPERIMENT LOOP ==============

all_results = []
all_metadata = []
all_model_performance = []
start_time = time.time()
datasets_completed = 0

logger.info("")
logger.info("="*70)
logger.info("STARTING EXPERIMENTS")
logger.info("="*70)

for dataset_idx, dataset_id in enumerate(DATASET_IDS):
    
    # Check for existing checkpoint
    _ckpt_key = f'session1_dataset_{dataset_id}'
    _ckpt = load_checkpoint(CHECKPOINTS_DIR, _ckpt_key)

    if _ckpt is not None:
        logger.info(f"\n[{dataset_idx+1}/{len(DATASET_IDS)}] Dataset {dataset_id} - SKIPPING (checkpoint exists)")
        all_results.extend(_ckpt.get('results', []))
        if 'metadata' in _ckpt and _ckpt['metadata']:
            all_metadata.append(_ckpt['metadata'])
        if 'model_performance' in _ckpt:
            all_model_performance.extend(_ckpt['model_performance'])
        datasets_completed += 1
        continue
    
    logger.info("")
    logger.info("="*70)
    logger.info(f"[{dataset_idx+1}/{len(DATASET_IDS)}] DATASET ID: {dataset_id}")
    logger.info("="*70)
    
    dataset_start = time.time()
    dataset_results = []
    dataset_model_perf = []
    dataset_metadata = None
    
    for seed_idx, seed in enumerate(CONFIG['random_seeds']):
        logger.info(f"\n  SEED {seed} ({seed_idx+1}/{len(CONFIG['random_seeds'])})")
        logger.info("  " + "-"*50)
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Load and preprocess
        result = load_and_preprocess(dataset_id, seed)
        X_train, X_test, y_train, y_test, feature_names, dataset_name, metadata = result
        
        if X_train is None:
            logger.error(f"  FAILED to load dataset - skipping seed")
            continue
        
        # Save metadata once
        if dataset_metadata is None:
            dataset_metadata = metadata
        
        logger.info(f"  Dataset: {dataset_name}")
        logger.info(f"  Size: {metadata['n_instances']} instances, {metadata['n_features']} features")
        logger.info(f"  Class balance: {metadata['class_balance']:.1%} minority")
        
        # Train models
        logger.info(f"  Training models...")
        models, train_times = train_models(X_train, y_train, seed)
        logger.info(f"  Trained: {len(models)}/{len(MODEL_NAMES)} models")
        
        if len(models) < 3:
            logger.warning(f"  Too few models - skipping seed")
            continue
        
        # Evaluate
        model_metrics = evaluate_models(models, X_test, y_test)
        acc_str = " | ".join([f"{k[:2].upper()}:{v['accuracy']:.2f}" for k, v in model_metrics.items()])
        logger.info(f"  Accuracy: {acc_str}")
        
        # Store performance
        for model_name, metrics in model_metrics.items():
            dataset_model_perf.append({
                'dataset_id': dataset_id,
                'dataset_name': dataset_name,
                'seed': seed,
                'model': model_name,
                'accuracy': metrics['accuracy'],
                'auc': metrics['auc'],
                'train_time': train_times.get(model_name, 0)
            })
        
        # Find agreement instances
        agreement_idx = find_agreement_instances(models, X_test, y_test)
        agreement_rate = len(agreement_idx) / len(y_test) if len(y_test) > 0 else 0
        logger.info(f"  Agreement: {len(agreement_idx)}/{len(y_test)} instances ({agreement_rate:.1%})")
        
        if len(agreement_idx) < CONFIG['min_agreement_instances']:
            logger.warning(f"  Too few agreement instances - skipping seed")
            continue
        
        # Sample if needed
        if len(agreement_idx) > CONFIG['max_instances_per_dataset']:
            agreement_idx = np.random.choice(
                agreement_idx, 
                CONFIG['max_instances_per_dataset'], 
                replace=False
            )
            logger.info(f"  Sampled: {len(agreement_idx)} instances")
        
        # Compute SHAP values
        logger.info(f"  Computing SHAP values...")
        shap_start = time.time()
        X_agreement = X_test[agreement_idx]
        shap_dict = {}
        
        for name, model in models.items():
            shap_vals = get_shap_values(model, X_agreement, X_train, name)
            if shap_vals is not None:
                shap_dict[name] = shap_vals
                logger.info(f"    {name}: OK")
            else:
                logger.warning(f"    {name}: FAILED")
        
        shap_time = time.time() - shap_start
        logger.info(f"  SHAP done: {len(shap_dict)} models ({shap_time:.1f}s)")
        
        if len(shap_dict) < 2:
            logger.warning(f"  Too few SHAP results - skipping seed")
            continue
        
        # Compute agreement metrics
        logger.info(f"  Computing metrics...")
        metrics_start = time.time()
        n_comparisons = 0
        
        for local_idx in range(len(agreement_idx)):
            metrics = compute_agreement_metrics(shap_dict, local_idx, len(feature_names))
            
            for m in metrics:
                m['dataset_id'] = dataset_id
                m['dataset_name'] = dataset_name
                m['seed'] = seed
                m['instance_idx'] = int(agreement_idx[local_idx])
                m['n_features'] = len(feature_names)
                m['n_instances'] = metadata['n_instances']
                m['agreement_rate'] = round(agreement_rate, 4)
                dataset_results.append(m)
                n_comparisons += 1
        
        metrics_time = time.time() - metrics_start
        logger.info(f"  Metrics done: {n_comparisons} comparisons ({metrics_time:.1f}s)")
    
    # Save checkpoint
    if len(dataset_results) > 0:
        all_results.extend(dataset_results)
        all_model_performance.extend(dataset_model_perf)
        if dataset_metadata:
            all_metadata.append(dataset_metadata)
        
        checkpoint_data = {
            'dataset_id': dataset_id,
            'dataset_name': dataset_name if dataset_name else f"dataset_{dataset_id}",
            'completed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'n_results': len(dataset_results),
            'metadata': dataset_metadata,
            'model_performance': dataset_model_perf,
            'results': dataset_results
        }
        
        save_checkpoint(CHECKPOINTS_DIR, _ckpt_key, checkpoint_data)

        # Save intermediate results
        pd.DataFrame(all_results).to_csv(f'{RESULTS_DIR}/intermediate_results.csv', index=False)
        pd.DataFrame(all_metadata).to_csv(f'{RESULTS_DIR}/dataset_metadata.csv', index=False)
        pd.DataFrame(all_model_performance).to_csv(f'{RESULTS_DIR}/model_performance.csv', index=False)
        
        datasets_completed += 1
        dataset_time = (time.time() - dataset_start) / 60
        elapsed_total = time.time() - start_time
        time_remaining = estimate_time_remaining(elapsed_total, datasets_completed, len(DATASET_IDS))
        
        logger.info("")
        logger.info(f"  DATASET {dataset_id} COMPLETE")
        logger.info(f"  Time: {dataset_time:.1f} min | Total: {elapsed_total/60:.1f} min")
        logger.info(f"  Progress: {datasets_completed}/{len(DATASET_IDS)} | ETA: {time_remaining}")
        logger.info(f"  Results: {len(dataset_results)} rows | Total: {len(all_results)} rows")
    else:
        logger.warning(f"\n  NO RESULTS for dataset {dataset_id}")

# ============== STEP 8: SAVE FINAL RESULTS ==============

logger.info("")
logger.info("="*70)
logger.info("FINALIZING SESSION 1")
logger.info("="*70)

if len(all_results) > 0:
    results_df = pd.DataFrame(all_results)
    
    # Validate
    issues = validate_results(results_df)
    if issues:
        logger.warning(f"Validation issues: {issues}")
    else:
        logger.info("Validation: PASSED")
    
    # Save files
    results_df.to_csv(f'{RESULTS_DIR}/final_results_session1.csv', index=False)
    logger.info(f"Saved: {RESULTS_DIR}/final_results_session1.csv")
    
    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_csv(f'{RESULTS_DIR}/dataset_metadata.csv', index=False)
    logger.info(f"Saved: {RESULTS_DIR}/dataset_metadata.csv")
    
    perf_df = pd.DataFrame(all_model_performance)
    perf_df.to_csv(f'{RESULTS_DIR}/model_performance.csv', index=False)
    logger.info(f"Saved: {RESULTS_DIR}/model_performance.csv")
    
    # Compute summary statistics
    total_time = (time.time() - start_time) / 60
    overall_spearman = results_df['spearman'].mean()
    overall_std = results_df['spearman'].std()
    
    logger.info("")
    logger.info("="*70)
    logger.info("SESSION 1 SUMMARY")
    logger.info("="*70)
    logger.info(f"Total time: {total_time:.1f} min ({total_time/60:.1f} hours)")
    logger.info(f"Datasets: {results_df['dataset_id'].nunique()}")
    logger.info(f"Total comparisons: {len(results_df):,}")
    logger.info(f"Overall Spearman: {overall_spearman:.4f} +/- {overall_std:.4f}")
    
    # Model pair analysis
    logger.info("")
    logger.info("SPEARMAN BY MODEL PAIR:")
    pair_stats = results_df.groupby(['model_a', 'model_b'])['spearman'].agg(['mean', 'std', 'count'])
    for idx, row in pair_stats.iterrows():
        logger.info(f"  {idx[0]:18} vs {idx[1]:18}: {row['mean']:.3f} +/- {row['std']:.3f} (n={int(row['count'])})")
    
    # Top-3 overlap
    if 'top_3_overlap' in results_df.columns:
        top3_mean = results_df['top_3_overlap'].mean()
        logger.info(f"\nTop-3 Overlap: {top3_mean:.3f} ({top3_mean*100:.1f}%)")
    
    # Save summary JSON
    summary = {
        'session': 1,
        'version': '1.0_final',
        'completed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_time_minutes': round(total_time, 2),
        'n_datasets': int(results_df['dataset_id'].nunique()),
        'n_results': len(results_df),
        'overall_spearman_mean': round(overall_spearman, 4),
        'overall_spearman_std': round(overall_std, 4),
        'top_3_overlap_mean': round(results_df['top_3_overlap'].mean(), 4) if 'top_3_overlap' in results_df.columns else None,
        'model_pairs': pair_stats['mean'].round(4).to_dict()
    }
    
    with open(f'{RESULTS_DIR}/session1_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSaved: {RESULTS_DIR}/session1_summary.json")

else:
    logger.error("NO RESULTS COLLECTED - CHECK ERRORS ABOVE")

logger.info("")
logger.info("="*70)
logger.info(f"SESSION 1 FINISHED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("="*70)
logger.info("")
logger.info("OUTPUT FILES:")
logger.info(f"  {RESULTS_DIR}/final_results_session1.csv")
logger.info(f"  {RESULTS_DIR}/dataset_metadata.csv")
logger.info(f"  {RESULTS_DIR}/model_performance.csv")
logger.info(f"  {RESULTS_DIR}/session1_summary.json")
logger.info(f"  {log_filename}")
logger.info("")
logger.info("NEXT: Run 02_session2.py")
logger.info("="*70)
