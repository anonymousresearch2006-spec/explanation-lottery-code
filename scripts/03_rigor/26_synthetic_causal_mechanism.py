import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy import stats
import shap
import json

def generate_causal_data(n_samples=1000, mechanism='additive'):
    np.random.seed(42)
    X = np.random.randn(n_samples, 10)
    
    if mechanism == 'additive':
        # Purely linear/additive: y = 2*x0 + 3*x1 - x2
        z = 2*X[:,0] + 3*X[:,1] - 1*X[:,2]
    else:
        # Purely non-linear/interaction: y = x0 * x1 + x2 * x3
        z = 5 * (X[:,0] * X[:,1]) + 5 * (X[:,2] * X[:,3])
        
    y = (z > np.median(z)).astype(int)
    return pd.DataFrame(X, columns=[f'f{i}' for i in range(10)]), pd.Series(y)

def run_causal_test():
    results = {}
    
    for mech in ['additive', 'interaction']:
        print(f"Testing mechanism: {mech}")
        X, y = generate_causal_data(mechanism=mech)
        
        X_train = X.iloc[:800]
        X_test = X.iloc[800:]
        y_train = y.iloc[:800]
        y_test = y.iloc[800:]
        
        seed_rhos = []
        for seed in [42, 43, 44]:
            rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=seed).fit(X_train, y_train)
            lr = LogisticRegression().fit(X_train, y_train)
            
            # Agreement set
            preds = pd.DataFrame({'rf': rf.predict(X_test), 'lr': lr.predict(X_test)}, index=X_test.index)
            agreed = preds[preds['rf'] == preds['lr']].index
            X_explain = X_test.loc[agreed].iloc[:20]
            
            if len(X_explain) == 0: continue

            # Kernel explainer for equality
            bg = shap.kmeans(X_train, 20)
            ex_rf = shap.KernelExplainer(rf.predict_proba, bg)
            ex_lr = shap.KernelExplainer(lr.predict_proba, bg)
            
            s_rf = safe_extract(ex_rf.shap_values(X_explain))
            s_lr = safe_extract(ex_lr.shap_values(X_explain))
            
            rhos = [stats.spearmanr(s_rf[i], s_lr[i])[0] for i in range(len(s_rf))]
            seed_rhos.append(np.mean(rhos))
        
        results[mech] = {
            "mean_rho": float(np.mean(seed_rhos)),
            "std_rho": float(np.std(seed_rhos))
        }
    
    with open('causal_logic_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\n--- ELITE UPGRADE 3 SUMMARY (Causal Proof) ---")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    run_causal_test()
