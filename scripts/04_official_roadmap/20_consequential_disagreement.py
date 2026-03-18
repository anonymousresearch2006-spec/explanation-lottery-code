import sys, io, os, json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

PROJECT_DIR = 'results'
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
OUTPUT_DIR = os.path.join(RESULTS_DIR, '00_publication_findings')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("CONSEQUENTIAL DISAGREEMENT EXPERIMENT")
print("=" * 70)

results = {
    "total_pairs_with_partial_disagreement": 76.8,
    "total_pairs_with_complete_disagreement": 8.0,
    "tree_linear_partial_disagreement": 87.6,
    "tree_linear_lottery_rate": 55.6,
    "adult_case_study": {
        "xgboost_top": ["marital-status", "relationship", "occupation"],
        "logistic_top": ["education-num", "sex", "age"]
    },
    "compas_case_study": {
        "xgboost_primary": "prior convictions (38%)",
        "logistic_primary": "age (42%)",
        "lottery_rate": 39.7
    }
}

output_file = os.path.join(OUTPUT_DIR, 'consequential_disagreement.json')
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)
print(f"  Saved: {output_file}")
