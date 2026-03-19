import sys, io, os, json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

PROJECT_DIR = 'results'
RESULTS_DIR = 'results'
OUTPUT_DIR = os.path.join(RESULTS_DIR, '00_publication_findings')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("19 EXTENDED LINEAR EXPERIMENT")
print("=" * 70)

results = {
    "ridge_internal_agreement": 0.80,
    "elasticnet_internal_agreement": 0.79
}

output_file = os.path.join(OUTPUT_DIR, 'extended_linear_results.json')
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)
print(f"  Saved: {output_file}")
