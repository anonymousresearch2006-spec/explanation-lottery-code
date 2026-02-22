import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
from scipy import stats
import statsmodels.api as sm
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

def run_multivariate_unification():
    instance_results = []
    dataset_summary = []
    
    for d_info in DATASETS:
        print(f"Processing {d_info['name']}...")
        try:
            dataset = fetch_openml(data_id=d_info['id'], as_frame=True, parser='auto')
            X = dataset.data.select_dtypes(include=[np.number]).dropna()
            y = dataset.target.loc[X.index]
            y = (y == y.unique()[0]).astype(int)
            
            if len(X) > 400:
                X = X.sample(400, random_state=42)
                y = y.loc[X.index]
            
            # Feature complexity metrics
            dim = X.shape[1]
            redundancy = np.mean(np.abs(X.corr().values[np.triu_indices(dim, k=1)])) if dim > 1 else 0
            
            # Models
            rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42).fit(X, y)
            mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42).fit(X, y)
            
            # Predict
            preds = pd.DataFrame({'rf': rf.predict(X), 'mlp': mlp.predict(X)}, index=X.index)
            agreed = preds[preds['rf'] == preds['mlp']].index
            
            if len(agreed) < 20: continue
            
            X_explain = X.loc[agreed].iloc[:20]
            
            # TreeSHAP for Interactions
            explainer_rf = shap.TreeExplainer(rf)
            inter_vals = explainer_rf.shap_interaction_values(X_explain)
            if isinstance(inter_vals, list): inter_vals = inter_vals[1]
            if len(inter_vals.shape) == 4: inter_vals = inter_vals[:,:,:,1] # handle (n, f, f, 2)
            
            # KernelSHAP for MLP
            bg = shap.kmeans(X, 10)
            explainer_mlp = shap.KernelExplainer(mlp.predict_proba, bg)
            shap_mlp = explainer_mlp.shap_values(X_explain)
            if isinstance(shap_mlp, list): shap_mlp = shap_mlp[1]
            elif len(shap_mlp.shape) == 3: shap_mlp = shap_mlp[:,:,1]
            
            # Instance-level Loop
            for i in range(len(X_explain)):
                try:
                    # Cross-Model Disagreement (1 - Spearman)
                    # Main effects are the diagonal of the interaction matrix
                    shap_rf_local = np.diagonal(inter_vals[i]) 
                    
                    if len(shap_rf_local) != len(shap_mlp[i]): continue
                    
                    rho = stats.spearmanr(shap_rf_local, shap_mlp[i])[0]
                    if np.isnan(rho): continue
                    disagreement = 1.0 - rho
                    
                    # Interaction Mass
                    total_mass = np.abs(inter_vals[i]).sum()
                    diag_mass = np.abs(np.diagonal(inter_vals[i])).sum()
                    interaction_mass = (total_mass - diag_mass) / (total_mass + 1e-9)
                    
                    instance_results.append({
                        "disagreement": disagreement,
                        "interaction_mass": interaction_mass,
                        "dimensionality": dim,
                        "redundancy": redundancy
                    })
                except Exception as ex:
                    continue
            
            print(f"  Completed {d_info['name']}")
            
        except Exception as e:
            print(f"Error on {d_info['name']}: {e}")

    # Multivariate Regression
    df = pd.DataFrame(instance_results).dropna()
    print(f"\nFinal record count: {len(df)}")
    
    X_reg = df[['interaction_mass', 'dimensionality', 'redundancy']]
    X_reg = sm.add_constant(X_reg)
    y_reg = df['disagreement']
    
    model = sm.OLS(y_reg, X_reg).fit()
    print(model.summary())
    
    # Monotonicity by Quartile
    df['quartile'] = pd.qcut(df['interaction_mass'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    monotonicity = df.groupby('quartile')['disagreement'].mean().to_dict()
    
    output = {
        "regression": {
            "r2": model.rsquared,
            "params": model.params.to_dict(),
            "pvalues": model.pvalues.to_dict(),
            "tvalues": model.tvalues.to_dict()
        },
        "monotonicity": monotonicity,
        "n_samples": len(df)
    }
    
    with open('elegant_unification.json', 'w') as f:
        json.dump(output, f, indent=4)
        
    print("\n--- LEVEL 9.5 UNIFICATION COMPLETE ---")
    print(f"Interaction Coefficient: {model.params['interaction_mass']:.4f} (p={model.pvalues['interaction_mass']:.4f})")
    print(f"Dimensionality Coefficient: {model.params['dimensionality']:.4f} (p={model.pvalues['dimensionality']:.4f})")
    print(f"Monotonicity: {monotonicity}")

if __name__ == "__main__":
    run_multivariate_unification()
