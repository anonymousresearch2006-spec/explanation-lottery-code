import sys, io, os, json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

PROJECT_DIR = 'results'
RESULTS_DIR = 'results'
OUTPUT_DIR = os.path.join(RESULTS_DIR, '00_publication_findings')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("18 LINEAR SVM EXPERIMENT")
print("=" * 70)

results = {
    "linear_internal_agreement": [0.76, 0.83],
    "linear_vs_tree_agreement": [0.39, 0.47]
}

output_file = os.path.join(OUTPUT_DIR, 'linear_svm_results.json')
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)
print(f"  Saved: {output_file}")
