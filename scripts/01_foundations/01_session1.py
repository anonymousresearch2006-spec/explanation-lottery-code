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

# ============== STEP 3: SETUP PROJECT STRUCTURE ==============
PROJECT_DIR = "explanation_lottery"
RESULTS_DIR = f"{PROJECT_DIR}/results/session1"
CHECKPOINTS_DIR = f"{PROJECT_DIR}/checkpoints"
LOGS_DIR = f"{PROJECT_DIR}/logs"
DATA_DIR = f"{PROJECT_DIR}/data"

for directory in [PROJECT_DIR, RESULTS_DIR, CHECKPOINTS_DIR, LOGS_DIR, DATA_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============== STEP 4: SETUP LOGGING ==============
log_filename = f"{LOGS_DIR}/session1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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

# ============== STEP 6: HELPER FUNCTIONS ==============

def load_and_preprocess(dataset_id, random_state):
    """
    Load dataset from OpenML and preprocess for ML.
    """
    try:
        dataset = openml.datasets.get_dataset(dataset_id, download_data=True)
        X, y, _, feature_names = dataset.get_data(target=dataset.default_target_attribute)
        
        if X is None or y is None:
            logger.error(f"Dataset {dataset_id} returned None")
            return None, None, None, None, None, None, None
        
        X = pd.DataFrame(X, columns=feature_names)
        y = pd.Series(y, name='target')
        
        # Store original info
        n_original = len(X)
        n_features_original = len(feature_names)
        
        # Encode categorical columns
        categorical_cols = []
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                categorical_cols.append(col)
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Convert to numeric
        X = X.apply(pd.to_numeric, errors='coerce')
        
        # Handle missing values
        n_missing = X.isna().sum().sum()
        if X.isna().any().any():
            X = X.fillna(X.median())
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # Encode target
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y.astype(str))
        n_classes = len(np.unique(y_encoded))
        
        if n_classes != 2:
            logger.warning(f"Dataset {dataset_id} has {n_classes} classes, expected 2")
        
        class_balance = np.min(np.bincount(y_encoded)) / len(y_encoded)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y_encoded, 
            test_size=CONFIG['test_size'], 
            random_state=random_state, 
            stratify=y_encoded
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Metadata for paper
        metadata = {
            'dataset_id': dataset_id,
            'dataset_name': dataset.name,
            'n_instances': n_original,
            'n_features': n_features_original,
            'n_categorical': len(categorical_cols),
            'n_missing_values': int(n_missing),
            'n_classes': n_classes,
            'class_balance': round(class_balance, 4),
            'n_train': len(X_train_scaled),
            'n_test': len(X_test_scaled)
        }
        
        return X_train_scaled, X_test_scaled, y_train, y_test, list(X.columns), dataset.name, metadata
    
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_id}: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None, None, None, None, None, None


def train_models(X_train, y_train, random_state):
    """
    Train all 5 models with consistent hyperparameters.
    """
    models = {}
    train_times = {}
    params = CONFIG['model_params']
    
    # XGBoost
    try:
        start = time.time()
        model = XGBClassifier(
            n_estimators=params['n_estimators'], 
            max_depth=params['max_depth'], 
            learning_rate=params['learning_rate'],
            random_state=random_state, 
            eval_metric='logloss', 
            verbosity=0
        )
        model.fit(X_train, y_train)
        models['xgboost'] = model
        train_times['xgboost'] = round(time.time() - start, 2)
    except Exception as e:
        logger.warning(f"XGBoost failed: {str(e)[:100]}")
    
    # LightGBM
    try:
        start = time.time()
        model = LGBMClassifier(
            n_estimators=params['n_estimators'], 
            max_depth=params['max_depth'], 
            learning_rate=params['learning_rate'],
            random_state=random_state, 
            verbose=-1,
            force_col_wise=True
        )
        model.fit(X_train, y_train)
        models['lightgbm'] = model
        train_times['lightgbm'] = round(time.time() - start, 2)
    except Exception as e:
        logger.warning(f"LightGBM failed: {str(e)[:100]}")
    
    # CatBoost
    try:
        start = time.time()
        model = CatBoostClassifier(
            iterations=params['n_estimators'], 
            depth=params['max_depth'], 
            learning_rate=params['learning_rate'],
            random_seed=random_state, 
            verbose=False
        )
        model.fit(X_train, y_train)
        models['catboost'] = model
        train_times['catboost'] = round(time.time() - start, 2)
    except Exception as e:
        logger.warning(f"CatBoost failed: {str(e)[:100]}")
    
    # Random Forest
    try:
        start = time.time()
        model = RandomForestClassifier(
            n_estimators=params['n_estimators'], 
            max_depth=params['max_depth'], 
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        models['random_forest'] = model
        train_times['random_forest'] = round(time.time() - start, 2)
    except Exception as e:
        logger.warning(f"Random Forest failed: {str(e)[:100]}")
    
    # Logistic Regression
    try:
        start = time.time()
        model = LogisticRegression(
            max_iter=1000, 
            random_state=random_state,
            n_jobs=-1,
            solver='lbfgs'
        )
        model.fit(X_train, y_train)
        models['logistic_regression'] = model
        train_times['logistic_regression'] = round(time.time() - start, 2)
    except Exception as e:
        logger.warning(f"Logistic Regression failed: {str(e)[:100]}")
    
    return models, train_times


def evaluate_models(models, X_test, y_test):
    """
    Evaluate all models on test set.
    """
    results = {}
    
    for name, model in models.items():
        try:
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            results[name] = {
                'accuracy': round(accuracy_score(y_test, y_pred), 4),
                'auc': round(roc_auc_score(y_test, y_proba), 4)
            }
        except Exception as e:
            logger.warning(f"Evaluation failed for {name}: {str(e)[:50]}")
            results[name] = {'accuracy': 0.0, 'auc': 0.0}
    
    return results


def get_shap_values(model, X_explain, X_background, model_name):
    """
    Compute SHAP values for a model.
    
    CORRECTED: Uses proper background data for LinearExplainer.
    
    Args:
        model: trained model
        X_explain: instances to explain (test set subset)
        X_background: background data for baseline (training data sample)
        model_name: name of the model
    
    Returns:
        numpy array of absolute SHAP values, or None if failed
    """
    try:
        if model_name in ['xgboost', 'lightgbm', 'catboost', 'random_forest']:
            # TreeExplainer - optimized for tree models
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_explain)
        else:
            # LinearExplainer - needs background data for baseline
            # Sample background data if too large
            n_background = min(CONFIG['shap_background_samples'], len(X_background))
            if len(X_background) > n_background:
                indices = np.random.choice(len(X_background), n_background, replace=False)
                background_sample = X_background[indices]
            else:
                background_sample = X_background
            
            explainer = shap.LinearExplainer(model, background_sample)
            shap_values = explainer.shap_values(X_explain)
        
        # Handle multi-output format (binary classification)
        if isinstance(shap_values, list):
            # Take class 1 SHAP values
            shap_values = shap_values[1]
        
        # Handle 3D output (some models return 3D array)
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]
        
        # Return absolute values for importance magnitude
        return np.abs(shap_values)
    
    except Exception as e:
        logger.warning(f"SHAP failed for {model_name}: {str(e)[:100]}")
        return None


def find_agreement_instances(models, X_test, y_test):
    """
    Find test instances where ALL models predict correctly.
    """
    all_correct = np.ones(len(y_test), dtype=bool)
    
    for name, model in models.items():
        try:
            preds = model.predict(X_test)
            correct = (preds == y_test)
            all_correct &= correct
        except Exception as e:
            logger.warning(f"Prediction failed for {name}: {str(e)[:50]}")
            # If prediction fails, mark all as incorrect for this model
            all_correct &= False
    
    return np.where(all_correct)[0]


def compute_agreement_metrics(shap_dict, instance_idx, n_features):
    """
    Compute agreement metrics between all model pairs for one instance.
    
    Metrics computed:
    - Spearman rank correlation
    - Top-K overlap (K=3,5,10)
    - Top-K Jaccard similarity
    - Weighted correlation
    """
    results = []
    model_names = list(shap_dict.keys())
    
    for i, model_a in enumerate(model_names):
        for model_b in model_names[i+1:]:
            try:
                shap_a = shap_dict[model_a][instance_idx]
                shap_b = shap_dict[model_b][instance_idx]
                
                # Skip if invalid values
                if np.any(np.isnan(shap_a)) or np.any(np.isnan(shap_b)):
                    continue
                if np.all(shap_a == 0) or np.all(shap_b == 0):
                    continue
                
                # Compute rankings (descending by importance)
                rank_a = np.argsort(np.argsort(-shap_a))
                rank_b = np.argsort(np.argsort(-shap_b))
                
                # Spearman rank correlation
                if len(rank_a) > 1:
                    spearman_corr, spearman_pval = spearmanr(rank_a, rank_b)
                else:
                    spearman_corr, spearman_pval = 1.0, 0.0
                
                if np.isnan(spearman_corr):
                    spearman_corr = 0.0
                if np.isnan(spearman_pval):
                    spearman_pval = 1.0
                
                # Top-K overlap and Jaccard
                top_k_results = {}
                for k in CONFIG['top_k_values']:
                    if n_features >= k:
                        top_a = set(np.argsort(-shap_a)[:k])
                        top_b = set(np.argsort(-shap_b)[:k])
                        
                        # Overlap ratio
                        overlap = len(top_a & top_b) / k
                        top_k_results[f'top_{k}_overlap'] = round(overlap, 4)
                        
                        # Jaccard similarity
                        union = len(top_a | top_b)
                        jaccard = len(top_a & top_b) / union if union > 0 else 0
                        top_k_results[f'top_{k}_jaccard'] = round(jaccard, 4)
                
                # Weighted correlation by SHAP magnitude
                weights = (shap_a + shap_b) / 2
                weight_sum = weights.sum()
                if weight_sum > 0:
                    weights_norm = weights / weight_sum
                    weighted_corr = np.corrcoef(shap_a * weights_norm, shap_b * weights_norm)[0, 1]
                    if np.isnan(weighted_corr):
                        weighted_corr = 0.0
                else:
                    weighted_corr = 0.0
                
                # Cosine similarity
                norm_a = np.linalg.norm(shap_a)
                norm_b = np.linalg.norm(shap_b)
                if norm_a > 0 and norm_b > 0:
                    cosine_sim = np.dot(shap_a, shap_b) / (norm_a * norm_b)
                else:
                    cosine_sim = 0.0
                
                results.append({
                    'model_a': model_a,
                    'model_b': model_b,
                    'spearman': round(spearman_corr, 4),
                    'spearman_pval': round(spearman_pval, 6),
                    'weighted_corr': round(weighted_corr, 4),
                    'cosine_similarity': round(cosine_sim, 4),
                    **top_k_results
                })
            
            except Exception as e:
                continue
    
    return results


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
    checkpoint_file = f'{CHECKPOINTS_DIR}/session1_dataset_{dataset_id}.json'
    
    if os.path.exists(checkpoint_file):
        logger.info(f"\n[{dataset_idx+1}/{len(DATASET_IDS)}] Dataset {dataset_id} - SKIPPING (checkpoint exists)")
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                all_results.extend(checkpoint_data.get('results', []))
                if 'metadata' in checkpoint_data and checkpoint_data['metadata']:
                    all_metadata.append(checkpoint_data['metadata'])
                if 'model_performance' in checkpoint_data:
                    all_model_performance.extend(checkpoint_data['model_performance'])
                datasets_completed += 1
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {str(e)[:50]}")
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
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)
        
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
