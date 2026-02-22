import pandas as pd
import numpy as np

# Load real data
df = pd.read_csv('results/combined_results.csv')

print("\n--- TABLE DATA FOR FIGURE 1: Lottery Rate vs Threshold ---")
print("| Threshold (tau) | Lottery Rate (%) | 95% Confidence Interval (±%) |")
print("| :--- | :--- | :--- |")
# These are the X and Y coordinates shown in Figure 1
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for t in thresholds:
    rate = (df['spearman'] < t).mean() * 100
    ci = 1.96 * np.sqrt((rate/100 * (1 - rate/100)) / len(df)) * 100
    print(f"| {t:.1f} | {rate:.2f}% | {ci:.2f}% |")

print("\n--- TABLE DATA FOR FIGURE 2: Agreement Distributions ---")
print("| Pair Type | Mean Spearman (rho) | Std Deviation | Lottery Rate (tau < 0.5) |")
print("| :--- | :--- | :--- | :--- |")
# Classification logic used in the plotting script
tree_models = {'catboost', 'lightgbm', 'random_forest', 'xgboost', 'gradient_boosting'}
linear_models = {'logistic_regression', 'ridge'}

def get_pair_type(row):
    m1, m2 = row['model_a'], row['model_b']
    if m1 in tree_models and m2 in tree_models:
        return 'tree-tree'
    if (m1 in tree_models and m2 in linear_models) or (m1 in linear_models and m2 in tree_models):
        return 'tree-linear'
    return 'other'

df['pair_type'] = df.apply(get_pair_type, axis=1)

for pt in ['tree-tree', 'tree-linear']:
    subset = df[df['pair_type'] == pt]['spearman']
    mean_val = subset.mean()
    std_val = subset.std()
    lottery_rate = (subset < 0.5).mean() * 100
    print(f"| {pt.capitalize()} | {mean_val:.4f} | {std_val:.4f} | {lottery_rate:.2f}% |")
