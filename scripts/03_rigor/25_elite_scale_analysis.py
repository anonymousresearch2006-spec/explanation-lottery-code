import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy import stats
import shap
from sklearn.datasets import fetch_openml
import warnings
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Dataset configuration from previous rigor experiments
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

def run_elite_scale():
    results = []
    calibration_data = []

    for d_info in DATASETS:
        print(f"Processing {d_info['name']} (ID: {d_info['id']})...")
        try:
            dataset = fetch_openml(data_id=d_info['id'], as_frame=True, parser='auto')
            X = dataset.data.select_dtypes(include=[np.number]).dropna()
            y = dataset.target.loc[X.index]
            y = (y == y.unique()[0]).astype(int)
            y = pd.Series(y, index=X.index)
            
            if len(X) > 1000:
                X = X.sample(1000, random_state=42)
                y = y.loc[X.index]
                
            X_train = X.iloc[:int(len(X)*0.8)]
            X_test = X.iloc[int(len(X)*0.8):]
            y_train = y.iloc[:int(len(y)*0.8)]
            y_test = y.iloc[int(len(y)*0.8):]

            # Models
            rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42).fit(X_train, y_train)
            gb = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42).fit(X_train, y_train)
            lr = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)
            mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42).fit(X_train, y_train)

            # Agreement Set (instances where all models agree on prediction)
            preds = pd.DataFrame({
                'rf': rf.predict(X_test),
                'gb': gb.predict(X_test),
                'lr': lr.predict(X_test),
                'mlp': mlp.predict(X_test)
            }, index=X_test.index)
            
            # Agreement Set for MLP vs Committee (at least 2 models)
            # We relax this to (MLP == RF) OR (MLP == LR) etc. to ensure we get data
            agree_rf = preds[preds['rf'] == preds['mlp']].index
            agree_gb = preds[preds['gb'] == preds['mlp']].index
            agree_lr = preds[preds['lr'] == preds['mlp']].index
            
            union_agree = agree_rf.union(agree_gb).union(agree_lr)
            
            if len(union_agree) < 5:
                print(f"Skipping {d_info['name']} - too small agreement set.")
                continue

            X_explain = X_test.loc[union_agree].iloc[:20] 

            # SHAP Computation
            bg = shap.kmeans(X_train, 20)
            
            def get_shap(model, X):
                explainer = shap.KernelExplainer(model.predict_proba, bg)
                sv = explainer.shap_values(X)
                # Binary classification robust extraction
                if isinstance(sv, list):
                    return sv[1]
                if len(sv.shape) == 3:
                    return sv[:,:,1]
                return sv

            shap_rf = get_shap(rf, X_explain)
            shap_gb = get_shap(gb, X_explain)
            shap_lr = get_shap(lr, X_explain)
            shap_mlp = get_shap(mlp, X_explain)

            # Pairwise Correlations
            def avg_rho(s1, s2):
                rhos = [stats.spearmanr(s1[i], s2[i])[0] for i in range(len(s1))]
                return np.nanmean(rhos)

            agreement_matrix = {
                'tree_tree': avg_rho(shap_rf, shap_gb),
                'tree_linear': avg_rho(shap_rf, shap_lr),
                'deep_tree': avg_rho(shap_mlp, shap_rf),
                'deep_linear': avg_rho(shap_mlp, shap_lr)
            }
            
            print(f"  Results: {agreement_matrix}")
            results.append({"dataset": d_info['name'], **agreement_matrix})

            # Reliability Calibration (Upgrade 4)
            # R(x) = avg agreement between {RF, GB, LR}
            # Predict agreement with {MLP}
            for i in range(len(X_explain)):
                rho_committee = [
                    stats.spearmanr(shap_rf[i], shap_gb[i])[0],
                    stats.spearmanr(shap_rf[i], shap_lr[i])[0],
                    stats.spearmanr(shap_gb[i], shap_lr[i])[0]
                ]
                rx = np.mean(rho_committee)
                
                # Ground truth: MLP agreement with the committee mean
                rho_heldout = np.mean([
                    stats.spearmanr(shap_mlp[i], shap_rf[i])[0],
                    stats.spearmanr(shap_mlp[i], shap_gb[i])[0]
                ])
                
                calibration_data.append({'rx': float(rx), 'heldout': float(rho_heldout)})

        except Exception as e:
            print(f"Error on {d_info['name']}: {e}")

    # Aggregation
    df_results = pd.DataFrame(results)
    summary = df_results.mean(numeric_only=True).to_dict()
    std_devs = df_results.std(numeric_only=True).to_dict()
    
    # Statistical Significance (Paired Wilcoxon: Deep-Linear vs Deep-Tree)
    # We test if the Geometric Gap (DL > DT) is consistent across datasets
    stat, p_val = stats.wilcoxon(df_results['deep_linear'], df_results['deep_tree'], alternative='greater')
    
    # Quantitative Calibration Correlation
    calib_df = pd.DataFrame(calibration_data).dropna()
    calib_corr, calib_p = stats.spearmanr(calib_df['rx'], calib_df['heldout'])
    
    # Sanitize NaN values for JSON compliance (convert to null)
    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(x) for x in obj]
        elif isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    final_output = sanitize(final_output)

    with open('elite_results.json', 'w') as f:
        json.dump(final_output, f, indent=4)
        
    print("\n--- ELITE UPGRADE 1 & 4 SUMMARY ---")
    print(summary)
    
if __name__ == "__main__":
    run_elite_scale()
