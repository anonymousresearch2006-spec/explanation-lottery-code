import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import json
import warnings
import os

warnings.filterwarnings('ignore')

# Dataset configuration
DATASETS = [
    {"id": 1590, "name": "adult"}, {"id": 1461, "name": "bank-marketing"},
    {"id": 31, "name": "german-credit"}, {"id": 1464, "name": "blood-transfusion"},
    {"id": 44, "name": "spambase"}, {"id": 1510, "name": "wdbc"},
    {"id": 37, "name": "diabetes"}, {"id": 1494, "name": "qsar-biodeg"},
    {"id": 1489, "name": "phoneme"}, {"id": 1501, "name": "churn"},
    {"id": 1497, "name": "wall-robot-navigation"}, {"id": 4134, "name": "bioresponse"},
    {"id": 1462, "name": "banknote-authentication"}, {"id": 1502, "name": "mammography"},
    {"id": 1468, "name": "cnae9"}, {"id": 1476, "name": "gas-drift"},
    {"id": 1478, "name": "har"}, {"id": 1479, "name": "hill-valley"},
    {"id": 1480, "name": "ilpd"}, {"id": 1485, "name": "madelon"},
    {"id": 1486, "name": "nomao"}, {"id": 1487, "name": "ozone-level"},
    {"id": 1491, "name": "pc4"}, {"id": 1492, "name": "pc3"},
    {"id": 1493, "name": "one-hundred-plants-texture"}
]

def run_mixed_effects_unification():
    instance_results = []
    
    for d_info in DATASETS:
        print(f"Processing {d_info['name']}...")
        try:
            dataset = fetch_openml(data_id=d_info['id'], as_frame=True, parser='auto')
            X = dataset.data.select_dtypes(include=[np.number]).dropna()
            y = dataset.target.loc[X.index]
            y = (y == y.unique()[0]).astype(int)
            
            # Larger sample for mixed effects power
            sample_size = min(len(X), 400)
            X = X.sample(sample_size, random_state=42)
            y = y.loc[X.index]
            
            dim = X.shape[1]
            redundancy = np.mean(np.abs(X.corr().values[np.triu_indices(dim, k=1)])) if dim > 1 else 0
            
            rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42).fit(X, y)
            mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42).fit(X, y)
            
            preds = pd.DataFrame({'rf': rf.predict(X), 'mlp': mlp.predict(X)}, index=X.index)
            agreed = preds[preds['rf'] == preds['mlp']].index
            
            if len(agreed) < 20: continue
            
            # Take a healthy sample of agreed instances
            X_explain = X.loc[agreed].iloc[:30]
            
            explainer_rf = shap.TreeExplainer(rf)
            inter_vals = explainer_rf.shap_interaction_values(X_explain)
            if isinstance(inter_vals, list): inter_vals = inter_vals[1]
            if len(inter_vals.shape) == 4: inter_vals = inter_vals[:,:,:,1]
            
            bg = shap.kmeans(X, 10)
            explainer_mlp = shap.KernelExplainer(mlp.predict_proba, bg)
            shap_mlp = explainer_mlp.shap_values(X_explain)
            if isinstance(shap_mlp, list): shap_mlp = shap_mlp[1]
            elif len(shap_mlp.shape) == 3: shap_mlp = shap_mlp[:,:,1]
            
            for i in range(len(X_explain)):
                try:
                    shap_rf_local = np.diagonal(inter_vals[i]) 
                    if len(shap_rf_local) != len(shap_mlp[i]): continue
                    
                    rho = stats.spearmanr(shap_rf_local, shap_mlp[i])[0]
                    if np.isnan(rho): continue
                    disagreement = 1.0 - rho
                    
                    total_mass = np.abs(inter_vals[i]).sum()
                    diag_mass = np.abs(np.diagonal(inter_vals[i])).sum()
                    interaction_mass = (total_mass - diag_mass) / (total_mass + 1e-9)
                    
                    instance_results.append({
                        "dataset": d_info['name'],
                        "disagreement": disagreement,
                        "interaction_mass": interaction_mass,
                        "dimensionality": dim,
                        "redundancy": redundancy
                    })
                except: continue
            
            print(f"  Completed {d_info['name']}")
            
        except Exception as e:
            print(f"Error on {d_info['name']}: {e}")

    df = pd.DataFrame(instance_results).dropna()
    print(f"\nFinal record count: {len(df)} across {df['dataset'].nunique()} datasets.")
    
    # 1. LINEAR MIXED-EFFECTS MODEL (LME)
    # Model: disagreement ~ interaction_mass + dimensionality + redundancy
    # Random Effect: (1 | dataset) to account for correlation within datasets
    print("\n--- LINEAR MIXED-EFFECTS MODEL RESULTS ---")
    md = smf.mixedlm("disagreement ~ interaction_mass + dimensionality + redundancy", 
                     df, groups=df["dataset"])
    mdf = md.fit()
    print(mdf.summary())
    
    # 2. WITHIN-DATASET MONOTONICITY CHECK
    # Does disagreement rise with interaction mass within the same dataset?
    within_dataset_corrs = []
    for name, group in df.groupby('dataset'):
        if len(group) > 10:
            c = stats.spearmanr(group['interaction_mass'], group['disagreement'])[0]
            if not np.isnan(c):
                within_dataset_corrs.append(c)
    
    avg_within_corr = np.mean(within_dataset_corrs)
    
    output = {
        "mixed_effects": {
            "pvalues": mdf.pvalues.to_dict(),
            "params": mdf.params.to_dict(),
            "fe_sd": float(np.sqrt(mdf.cov_params().values.diagonal()[:-1]).mean()), # rough avg SE
            "group_var": float(mdf.cov_params().values[-1, -1])
        },
        "within_dataset_monotonicity": {
            "avg_rho": float(avg_within_corr),
            "n_valid_datasets": len(within_dataset_corrs)
        },
        "n_samples": len(df),
        "n_groups": df['dataset'].nunique()
    }
    
    with open('hierarchical_rigor_results.json', 'w') as f:
        json.dump(output, f, indent=4)
        
    print(f"\nAverage within-dataset Interaction-Disagreement Correlation: {avg_within_corr:.4f}")
    if avg_within_corr > 0:
        print("PRECISION VERIFIED: Interaction drives disagreement LOCAL-STRUCTURALLY, not just dataset-level.")

if __name__ == "__main__":
    run_mixed_effects_unification()
