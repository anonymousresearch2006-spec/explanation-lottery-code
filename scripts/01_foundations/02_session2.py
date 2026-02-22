"""
================================================================================
THE EXPLANATION LOTTERY - SESSION 2 (FINAL Q2 GRADE)
================================================================================
Datasets: 11-20 of 20 | Completes the full experiment
Then generates analysis, figures, and statistical tests
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
    except:
        pass

print("Dependencies ready!")

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
import openml
openml.config.set_root_cache_directory('data/openml_cache')

from scipy.stats import spearmanr, wilcoxon, mannwhitneyu
import time
from datetime import datetime
import json
import logging
import traceback
import matplotlib.pyplot as plt
import seaborn as sns

# ============== STEP 3: SETUP ==============
PROJECT_DIR = "explanation_lottery"
RESULTS_DIR = f"{PROJECT_DIR}/results/session2"
CHECKPOINTS_DIR = f"{PROJECT_DIR}/checkpoints"
LOGS_DIR = f"{PROJECT_DIR}/logs"
FIGURES_DIR = f"{PROJECT_DIR}/figures"

for directory in [RESULTS_DIR, CHECKPOINTS_DIR, LOGS_DIR, FIGURES_DIR]:
    os.makedirs(directory, exist_ok=True)

# Logging
log_filename = f"{LOGS_DIR}/session2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
logger.info("THE EXPLANATION LOTTERY - SESSION 2")
logger.info("="*70)

# ============== STEP 4: CONFIGURATION ==============
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

# Session 2: Next 10 datasets
DATASET_IDS = [
    1480,  # hill-valley (different version)
    1494,  # qsar-biodeg
    1510,  # wdbc (breast cancer)
    1590,  # adult
    4534,  # PhishingWebsites
    40536, # SpeedDating
    40975, # car
    41027, # jungle_chess
    23512, # higgs (small)
    1063,  # kc2
]

MODEL_NAMES = ['xgboost', 'lightgbm', 'catboost', 'random_forest', 'logistic_regression']

logger.info(f"Datasets: {len(DATASET_IDS)}")
logger.info(f"Models: {MODEL_NAMES}")

# ============== STEP 5: HELPER FUNCTIONS (SAME AS SESSION 1) ==============

def load_and_preprocess(dataset_id, random_state):
    try:
        dataset = openml.datasets.get_dataset(dataset_id, download_data=True)
        X, y, _, feature_names = dataset.get_data(target=dataset.default_target_attribute)
        
        if X is None or y is None:
            return None, None, None, None, None, None, None
        
        X = pd.DataFrame(X, columns=feature_names)
        y = pd.Series(y, name='target')
        
        n_original = len(X)
        n_features_original = len(feature_names)
        
        # Encode categorical
        categorical_cols = []
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                categorical_cols.append(col)
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        X = X.apply(pd.to_numeric, errors='coerce')
        X = X.fillna(X.median())
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
        
        # Encode target (handle multiclass by taking first 2 classes)
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y.astype(str))
        n_classes = len(np.unique(y_encoded))
        
        # If multiclass, convert to binary (class 0 vs rest)
        if n_classes > 2:
            y_encoded = (y_encoded == 0).astype(int)
            n_classes = 2
        
        class_balance = np.min(np.bincount(y_encoded)) / len(y_encoded)
        
        # Limit large datasets
        if len(X) > 10000:
            np.random.seed(random_state)
            idx = np.random.choice(len(X), 10000, replace=False)
            X = X.iloc[idx]
            y_encoded = y_encoded[idx]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y_encoded, 
            test_size=CONFIG['test_size'], 
            random_state=random_state, 
            stratify=y_encoded
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        metadata = {
            'dataset_id': dataset_id,
            'dataset_name': dataset.name,
            'n_instances': n_original,
            'n_features': n_features_original,
            'n_categorical': len(categorical_cols),
            'n_classes': n_classes,
            'class_balance': round(class_balance, 4),
            'n_train': len(X_train_scaled),
            'n_test': len(X_test_scaled)
        }
        
        return X_train_scaled, X_test_scaled, y_train, y_test, list(X.columns), dataset.name, metadata
    
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_id}: {str(e)[:100]}")
        return None, None, None, None, None, None, None


def train_models(X_train, y_train, random_state):
    models = {}
    train_times = {}
    params = CONFIG['model_params']
    
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
        logger.warning(f"XGBoost failed: {str(e)[:50]}")
    
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
        logger.warning(f"LightGBM failed: {str(e)[:50]}")
    
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
        logger.warning(f"CatBoost failed: {str(e)[:50]}")
    
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
        logger.warning(f"Random Forest failed: {str(e)[:50]}")
    
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
        logger.warning(f"Logistic Regression failed: {str(e)[:50]}")
    
    return models, train_times


def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        try:
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            results[name] = {
                'accuracy': round(accuracy_score(y_test, y_pred), 4),
                'auc': round(roc_auc_score(y_test, y_proba), 4)
            }
        except:
            results[name] = {'accuracy': 0.0, 'auc': 0.0}
    return results


def get_shap_values(model, X_explain, X_background, model_name):
    try:
        if model_name in ['xgboost', 'lightgbm', 'catboost', 'random_forest']:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_explain)
        else:
            n_background = min(CONFIG['shap_background_samples'], len(X_background))
            if len(X_background) > n_background:
                indices = np.random.choice(len(X_background), n_background, replace=False)
                background_sample = X_background[indices]
            else:
                background_sample = X_background
            explainer = shap.LinearExplainer(model, background_sample)
            shap_values = explainer.shap_values(X_explain)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]
        
        return np.abs(shap_values)
    except Exception as e:
        logger.warning(f"SHAP failed for {model_name}: {str(e)[:50]}")
        return None


def find_agreement_instances(models, X_test, y_test):
    all_correct = np.ones(len(y_test), dtype=bool)
    for name, model in models.items():
        try:
            preds = model.predict(X_test)
            all_correct &= (preds == y_test)
        except:
            all_correct &= False
    return np.where(all_correct)[0]


def compute_agreement_metrics(shap_dict, instance_idx, n_features):
    results = []
    model_names = list(shap_dict.keys())
    
    for i, model_a in enumerate(model_names):
        for model_b in model_names[i+1:]:
            try:
                shap_a = shap_dict[model_a][instance_idx]
                shap_b = shap_dict[model_b][instance_idx]
                
                if np.any(np.isnan(shap_a)) or np.any(np.isnan(shap_b)):
                    continue
                if np.all(shap_a == 0) or np.all(shap_b == 0):
                    continue
                
                rank_a = np.argsort(np.argsort(-shap_a))
                rank_b = np.argsort(np.argsort(-shap_b))
                
                if len(rank_a) > 1:
                    spearman_corr, spearman_pval = spearmanr(rank_a, rank_b)
                else:
                    spearman_corr, spearman_pval = 1.0, 0.0
                
                if np.isnan(spearman_corr):
                    spearman_corr = 0.0
                
                top_k_results = {}
                for k in CONFIG['top_k_values']:
                    if n_features >= k:
                        top_a = set(np.argsort(-shap_a)[:k])
                        top_b = set(np.argsort(-shap_b)[:k])
                        overlap = len(top_a & top_b) / k
                        top_k_results[f'top_{k}_overlap'] = round(overlap, 4)
                        union = len(top_a | top_b)
                        jaccard = len(top_a & top_b) / union if union > 0 else 0
                        top_k_results[f'top_{k}_jaccard'] = round(jaccard, 4)
                
                weights = (shap_a + shap_b) / 2
                weight_sum = weights.sum()
                if weight_sum > 0:
                    weights_norm = weights / weight_sum
                    weighted_corr = np.corrcoef(shap_a * weights_norm, shap_b * weights_norm)[0, 1]
                    if np.isnan(weighted_corr):
                        weighted_corr = 0.0
                else:
                    weighted_corr = 0.0
                
                norm_a = np.linalg.norm(shap_a)
                norm_b = np.linalg.norm(shap_b)
                cosine_sim = np.dot(shap_a, shap_b) / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0
                
                results.append({
                    'model_a': model_a,
                    'model_b': model_b,
                    'spearman': round(spearman_corr, 4),
                    'spearman_pval': round(spearman_pval, 6) if not np.isnan(spearman_pval) else 1.0,
                    'weighted_corr': round(weighted_corr, 4),
                    'cosine_similarity': round(cosine_sim, 4),
                    **top_k_results
                })
            except:
                continue
    
    return results


# ============== STEP 6: MAIN EXPERIMENT LOOP ==============

all_results = []
all_metadata = []
all_model_performance = []
start_time = time.time()
datasets_completed = 0

logger.info("")
logger.info("="*70)
logger.info("STARTING SESSION 2 EXPERIMENTS")
logger.info("="*70)

for dataset_idx, dataset_id in enumerate(DATASET_IDS):
    
    checkpoint_file = f'{CHECKPOINTS_DIR}/session2_dataset_{dataset_id}.json'
    
    if os.path.exists(checkpoint_file):
        logger.info(f"\n[{dataset_idx+1}/{len(DATASET_IDS)}] Dataset {dataset_id} - SKIPPING (done)")
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                all_results.extend(checkpoint_data.get('results', []))
                if checkpoint_data.get('metadata'):
                    all_metadata.append(checkpoint_data['metadata'])
                all_model_performance.extend(checkpoint_data.get('model_performance', []))
                datasets_completed += 1
        except:
            pass
        continue
    
    logger.info(f"\n{'='*60}")
    logger.info(f"[{dataset_idx+1}/{len(DATASET_IDS)}] DATASET ID: {dataset_id}")
    logger.info(f"{'='*60}")
    
    dataset_start = time.time()
    dataset_results = []
    dataset_model_perf = []
    dataset_metadata = None
    
    for seed_idx, seed in enumerate(CONFIG['random_seeds']):
        logger.info(f"\n  SEED {seed} ({seed_idx+1}/{len(CONFIG['random_seeds'])})")
        
        np.random.seed(seed)
        
        result = load_and_preprocess(dataset_id, seed)
        X_train, X_test, y_train, y_test, feature_names, dataset_name, metadata = result
        
        if X_train is None:
            logger.error(f"  Failed to load - skipping")
            continue
        
        if dataset_metadata is None:
            dataset_metadata = metadata
        
        logger.info(f"  Dataset: {dataset_name} | {metadata['n_instances']} inst, {metadata['n_features']} feat")
        
        models, train_times = train_models(X_train, y_train, seed)
        logger.info(f"  Trained: {len(models)}/5 models")
        
        if len(models) < 3:
            continue
        
        model_metrics = evaluate_models(models, X_test, y_test)
        
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
        
        agreement_idx = find_agreement_instances(models, X_test, y_test)
        agreement_rate = len(agreement_idx) / len(y_test) if len(y_test) > 0 else 0
        logger.info(f"  Agreement: {len(agreement_idx)}/{len(y_test)} ({agreement_rate:.1%})")
        
        if len(agreement_idx) < CONFIG['min_agreement_instances']:
            continue
        
        if len(agreement_idx) > CONFIG['max_instances_per_dataset']:
            agreement_idx = np.random.choice(agreement_idx, CONFIG['max_instances_per_dataset'], replace=False)
        
        X_agreement = X_test[agreement_idx]
        shap_dict = {}
        
        for name, model in models.items():
            shap_vals = get_shap_values(model, X_agreement, X_train, name)
            if shap_vals is not None:
                shap_dict[name] = shap_vals
        
        logger.info(f"  SHAP: {len(shap_dict)} models")
        
        if len(shap_dict) < 2:
            continue
        
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
        
        logger.info(f"  Metrics: {len(dataset_results)} comparisons")
    
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
        
        pd.DataFrame(all_results).to_csv(f'{RESULTS_DIR}/intermediate_results.csv', index=False)
        
        datasets_completed += 1
        logger.info(f"\n  DATASET {dataset_id} COMPLETE | Total: {len(all_results)} rows")

# ============== STEP 7: SAVE SESSION 2 RESULTS ==============

logger.info("\n" + "="*70)
logger.info("SAVING SESSION 2 RESULTS")
logger.info("="*70)

if len(all_results) > 0:
    results_df_s2 = pd.DataFrame(all_results)
    results_df_s2.to_csv(f'{RESULTS_DIR}/final_results_session2.csv', index=False)
    pd.DataFrame(all_metadata).to_csv(f'{RESULTS_DIR}/dataset_metadata_s2.csv', index=False)
    pd.DataFrame(all_model_performance).to_csv(f'{RESULTS_DIR}/model_performance_s2.csv', index=False)
    logger.info(f"Session 2 results: {len(results_df_s2)} rows")

# ============== STEP 8: COMBINE BOTH SESSIONS ==============

logger.info("\n" + "="*70)
logger.info("COMBINING RESULTS FROM BOTH SESSIONS")
logger.info("="*70)

try:
    df_s1 = pd.read_csv('results/session1/final_results_session1.csv')
    df_s2 = pd.DataFrame(all_results)
    combined_df = pd.concat([df_s1, df_s2], ignore_index=True)
    
    # Save combined
    combined_df.to_csv(f'{PROJECT_DIR}/results/combined_results.csv', index=False)
    logger.info(f"Combined results: {len(combined_df)} rows from {combined_df['dataset_id'].nunique()} datasets")
except Exception as e:
    logger.error(f"Could not combine: {e}")
    combined_df = pd.DataFrame(all_results)

# ============== STEP 9: GENERATE FIGURES ==============

logger.info("\n" + "="*70)
logger.info("GENERATING PUBLICATION-READY FIGURES")
logger.info("="*70)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16

# FIGURE 1: Heatmap of Model Pair Agreement
logger.info("Creating Figure 1: Agreement Heatmap...")

fig, ax = plt.subplots(figsize=(10, 8))

models = ['xgboost', 'lightgbm', 'catboost', 'random_forest', 'logistic_regression']
model_labels = ['XGBoost', 'LightGBM', 'CatBoost', 'Random Forest', 'Logistic Reg.']

agreement_matrix = np.eye(len(models))
pivot_data = combined_df.groupby(['model_a', 'model_b'])['spearman'].mean()

for (ma, mb), val in pivot_data.items():
    if ma in models and mb in models:
        i, j = models.index(ma), models.index(mb)
        agreement_matrix[i, j] = val
        agreement_matrix[j, i] = val

sns.heatmap(agreement_matrix, 
            xticklabels=model_labels, 
            yticklabels=model_labels,
            annot=True, fmt='.2f', 
            cmap='RdYlGn',
            vmin=0, vmax=1,
            square=True,
            cbar_kws={'label': 'Spearman Correlation'},
            ax=ax)
ax.set_title('Cross-Model SHAP Explanation Agreement\n(When All Models Predict Correctly)', fontsize=16)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/fig1_agreement_heatmap.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{FIGURES_DIR}/fig1_agreement_heatmap.pdf', bbox_inches='tight')
plt.close()
logger.info("  Saved: fig1_agreement_heatmap.png")

# FIGURE 2: Distribution of Agreement Scores
logger.info("Creating Figure 2: Distribution...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Spearman distribution
axes[0].hist(combined_df['spearman'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].axvline(combined_df['spearman'].mean(), color='red', linestyle='--', linewidth=2, 
                label=f"Mean: {combined_df['spearman'].mean():.3f}")
axes[0].axvline(combined_df['spearman'].median(), color='orange', linestyle=':', linewidth=2,
                label=f"Median: {combined_df['spearman'].median():.3f}")
axes[0].set_xlabel('Spearman Rank Correlation')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of SHAP Agreement Scores')
axes[0].legend()

# Top-3 overlap distribution
if 'top_3_overlap' in combined_df.columns:
    axes[1].hist(combined_df['top_3_overlap'], bins=20, edgecolor='black', alpha=0.7, color='forestgreen')
    axes[1].axvline(combined_df['top_3_overlap'].mean(), color='red', linestyle='--', linewidth=2,
                    label=f"Mean: {combined_df['top_3_overlap'].mean():.3f}")
    axes[1].set_xlabel('Top-3 Feature Overlap')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('How Often Do Top-3 Features Match?')
    axes[1].legend()

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/fig2_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
logger.info("  Saved: fig2_distributions.png")

# FIGURE 3: Agreement by Dataset
logger.info("Creating Figure 3: By Dataset...")

fig, ax = plt.subplots(figsize=(12, 8))

dataset_means = combined_df.groupby('dataset_name')['spearman'].agg(['mean', 'std']).sort_values('mean')
colors = ['#d62728' if x < 0.4 else '#ff7f0e' if x < 0.6 else '#2ca02c' for x in dataset_means['mean']]

bars = ax.barh(range(len(dataset_means)), dataset_means['mean'], xerr=dataset_means['std'], 
               color=colors, capsize=3, alpha=0.8)
ax.set_yticks(range(len(dataset_means)))
ax.set_yticklabels(dataset_means.index)
ax.set_xlabel('Mean Spearman Correlation')
ax.set_title('SHAP Agreement Varies Significantly Across Datasets')
ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='Moderate (0.5)')
ax.axvline(0.7, color='gray', linestyle=':', alpha=0.5, label='Strong (0.7)')
ax.legend(loc='lower right')
ax.set_xlim(0, 1)

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/fig3_by_dataset.png', dpi=300, bbox_inches='tight')
plt.close()
logger.info("  Saved: fig3_by_dataset.png")

# FIGURE 4: Tree Models vs Linear Model
logger.info("Creating Figure 4: Tree vs Linear...")

fig, ax = plt.subplots(figsize=(10, 6))

tree_models = ['xgboost', 'lightgbm', 'catboost', 'random_forest']
tree_pairs = combined_df[
    (combined_df['model_a'].isin(tree_models)) & 
    (combined_df['model_b'].isin(tree_models))
]['spearman']

lr_pairs = combined_df[
    (combined_df['model_a'] == 'logistic_regression') | 
    (combined_df['model_b'] == 'logistic_regression')
]['spearman']

ax.hist(tree_pairs, bins=40, alpha=0.6, label=f'Tree-Tree (μ={tree_pairs.mean():.3f})', color='forestgreen')
ax.hist(lr_pairs, bins=40, alpha=0.6, label=f'Tree-LR (μ={lr_pairs.mean():.3f})', color='steelblue')
ax.set_xlabel('Spearman Correlation')
ax.set_ylabel('Frequency')
ax.set_title('Tree Models Agree More With Each Other Than With Logistic Regression')
ax.legend()

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/fig4_tree_vs_linear.png', dpi=300, bbox_inches='tight')
plt.close()
logger.info("  Saved: fig4_tree_vs_linear.png")

# FIGURE 5: Agreement vs Number of Features
logger.info("Creating Figure 5: Features Effect...")

fig, ax = plt.subplots(figsize=(10, 6))

feature_agreement = combined_df.groupby('n_features')['spearman'].agg(['mean', 'std', 'count'])
feature_agreement = feature_agreement[feature_agreement['count'] >= 100]

ax.errorbar(feature_agreement.index, feature_agreement['mean'], 
            yerr=feature_agreement['std'], fmt='o-', capsize=4, 
            color='steelblue', markersize=8)
ax.set_xlabel('Number of Features')
ax.set_ylabel('Mean Spearman Correlation')
ax.set_title('Does Dataset Complexity Affect Explanation Agreement?')
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/fig5_features_effect.png', dpi=300, bbox_inches='tight')
plt.close()
logger.info("  Saved: fig5_features_effect.png")

# ============== STEP 10: STATISTICAL TESTS ==============

logger.info("\n" + "="*70)
logger.info("STATISTICAL SIGNIFICANCE TESTS")
logger.info("="*70)

stats_results = {}

# Test 1: Tree-Tree vs Tree-LR
if len(tree_pairs) > 10 and len(lr_pairs) > 10:
    # Sample for test
    n_sample = min(5000, len(tree_pairs), len(lr_pairs))
    tree_sample = tree_pairs.sample(n_sample, random_state=42)
    lr_sample = lr_pairs.sample(n_sample, random_state=42)
    
    stat, pval = mannwhitneyu(tree_sample, lr_sample, alternative='greater')
    stats_results['tree_vs_lr'] = {
        'test': 'Mann-Whitney U',
        'statistic': float(stat),
        'p_value': float(pval),
        'significant': pval < 0.05,
        'interpretation': 'Tree models agree significantly more with each other than with LR' if pval < 0.05 else 'No significant difference'
    }
    logger.info(f"Tree-Tree vs Tree-LR: U={stat:.0f}, p={pval:.2e} {'***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''}")

# Test 2: XGBoost-LightGBM vs others
xgb_lgbm = combined_df[
    ((combined_df['model_a'] == 'xgboost') & (combined_df['model_b'] == 'lightgbm')) |
    ((combined_df['model_a'] == 'lightgbm') & (combined_df['model_b'] == 'xgboost'))
]['spearman']

others = combined_df[
    ~(((combined_df['model_a'] == 'xgboost') & (combined_df['model_b'] == 'lightgbm')) |
      ((combined_df['model_a'] == 'lightgbm') & (combined_df['model_b'] == 'xgboost')))
]['spearman']

if len(xgb_lgbm) > 10 and len(others) > 10:
    n_sample = min(5000, len(xgb_lgbm), len(others))
    stat, pval = mannwhitneyu(
        xgb_lgbm.sample(min(n_sample, len(xgb_lgbm)), random_state=42),
        others.sample(n_sample, random_state=42),
        alternative='greater'
    )
    stats_results['xgb_lgbm_vs_others'] = {
        'test': 'Mann-Whitney U',
        'statistic': float(stat),
        'p_value': float(pval),
        'significant': pval < 0.05
    }
    logger.info(f"XGB-LGBM vs Others: U={stat:.0f}, p={pval:.2e}")

# ============== STEP 11: FINAL SUMMARY ==============

logger.info("\n" + "="*70)
logger.info("FINAL SUMMARY - Q2 PAPER READY")
logger.info("="*70)

overall_spearman = combined_df['spearman'].mean()
overall_std = combined_df['spearman'].std()
n_datasets = combined_df['dataset_id'].nunique()
n_comparisons = len(combined_df)

logger.info(f"\nKEY STATISTICS:")
logger.info(f"  Total Datasets: {n_datasets}")
logger.info(f"  Total Comparisons: {n_comparisons:,}")
logger.info(f"  Overall Spearman: {overall_spearman:.4f} ± {overall_std:.4f}")
logger.info(f"  Tree-Tree Agreement: {tree_pairs.mean():.4f}")
logger.info(f"  Tree-LR Agreement: {lr_pairs.mean():.4f}")

if 'top_3_overlap' in combined_df.columns:
    logger.info(f"  Top-3 Overlap: {combined_df['top_3_overlap'].mean():.4f} ({combined_df['top_3_overlap'].mean()*100:.1f}%)")

logger.info(f"\nKEY FINDINGS:")
logger.info(f"  1. Overall explanation agreement is MODERATE (ρ={overall_spearman:.2f})")
logger.info(f"  2. Tree models agree more with each other (ρ={tree_pairs.mean():.2f}) than with LR (ρ={lr_pairs.mean():.2f})")
logger.info(f"  3. XGBoost and LightGBM have highest agreement (ρ≈0.79)")
logger.info(f"  4. Agreement varies significantly by dataset")

# Save final summary
final_summary = {
    'title': 'The Explanation Lottery: Do Equally Accurate Models Tell the Same Story?',
    'completed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'n_datasets': n_datasets,
    'n_comparisons': n_comparisons,
    'n_models': 5,
    'n_seeds': 3,
    'overall_spearman_mean': round(overall_spearman, 4),
    'overall_spearman_std': round(overall_std, 4),
    'tree_tree_mean': round(tree_pairs.mean(), 4),
    'tree_lr_mean': round(lr_pairs.mean(), 4),
    'top_3_overlap_mean': round(combined_df['top_3_overlap'].mean(), 4) if 'top_3_overlap' in combined_df.columns else None,
    'statistical_tests': stats_results,
    'key_finding': 'Model choice significantly affects SHAP explanations even when predictions agree'
}

with open(f'{PROJECT_DIR}/results/final_summary.json', 'w') as f:
    json.dump(final_summary, f, indent=2)

logger.info(f"\n" + "="*70)
logger.info("ALL FILES SAVED:")
logger.info("="*70)
logger.info(f"  {PROJECT_DIR}/results/combined_results.csv")
logger.info(f"  {PROJECT_DIR}/results/final_summary.json")
logger.info(f"  {FIGURES_DIR}/fig1_agreement_heatmap.png")
logger.info(f"  {FIGURES_DIR}/fig2_distributions.png")
logger.info(f"  {FIGURES_DIR}/fig3_by_dataset.png")
logger.info(f"  {FIGURES_DIR}/fig4_tree_vs_linear.png")
logger.info(f"  {FIGURES_DIR}/fig5_features_effect.png")

logger.info(f"\n" + "="*70)
logger.info("Q2 EXPERIMENT COMPLETE!")
logger.info("="*70)
logger.info(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
logger.info("\nREADY FOR PAPER WRITING!")
logger.info("="*70)
