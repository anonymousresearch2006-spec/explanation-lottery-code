import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_openml
from scipy import stats
import json
import warnings
import os

warnings.filterwarnings('ignore')

# Use datasets from elite results
with open('elite_results.json', 'r') as f:
    elite_data = json.load(f)
    dataset_rhos = {item['dataset']: item['deep_tree'] for item in elite_data['raw_per_dataset']}

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

def run_unification():
    unification_results = []
    
    for d_info in DATASETS:
        if d_info['name'] not in dataset_rhos or np.isnan(dataset_rhos[d_info['name']]):
            continue
            
        print(f"Analyzing interaction strength for {d_info['name']}...")
        try:
            dataset = fetch_openml(data_id=d_info['id'], as_frame=True, parser='auto')
            X = dataset.data.select_dtypes(include=[np.number]).dropna()
            y = dataset.target.loc[X.index]
            y = (y == y.unique()[0]).astype(int)
            
            if len(X) > 500:
                X = X.sample(500, random_state=42)
                y = y.loc[X.index]
            
            # Train RF for TreeSHAP Interaction Values
            rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42).fit(X, y)
            explainer = shap.TreeExplainer(rf)
            
            # Subsample for interaction computation (it's slow)
            X_sub = X.iloc[:20]
            inter_vals = explainer.shap_interaction_values(X_sub)
            
            if isinstance(inter_vals, list):
                inter_vals = inter_vals[1]
                
            # Global Interaction Strength (GIS)
            # Ratio of off-diagonal (interactions) to diagonal (main effects)
            main_effects = np.abs(np.diagonal(inter_vals, axis1=1, axis2=2)).mean()
            interactions = np.abs(inter_vals).mean() - (main_effects / inter_vals.shape[1])
            gis = interactions / (main_effects + 1e-9)
            
            unification_results.append({
                "dataset": d_info['name'],
                "interaction_strength": float(gis),
                "disagreement": 1.0 - dataset_rhos[d_info['name']]
            })
            print(f"  GIS: {gis:.4f}, Disagreement: {1.0 - dataset_rhos[d_info['name']]:.4f}")
            
        except Exception as e:
            print(f"Error on {d_info['name']}: {e}")

    df = pd.DataFrame(unification_results)
    if len(df) > 5:
        spearman_rho, p_val = stats.spearmanr(df['interaction_strength'], df['disagreement'])
        
        final_data = {
            "spearman_rho": float(spearman_rho),
            "p_value": float(p_val),
            "raw": unification_results
        }
        
        with open('unification_results.json', 'w') as f:
            json.dump(final_data, f, indent=4)
            
        print("\n--- ELEGANT UNIFICATION SUMMARY ---")
        print(f"Correlation (Interaction Strength vs Disagreement): {spearman_rho:.4f}")
        print(f"p-value: {p_val:.5f}")
    else:
        print("Not enough data for unification correlation.")

if __name__ == "__main__":
    run_unification()
