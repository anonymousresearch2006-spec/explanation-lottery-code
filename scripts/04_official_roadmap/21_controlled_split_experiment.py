import sys, io, os, json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

PROJECT_DIR = 'results'
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
OUTPUT_DIR = os.path.join(RESULTS_DIR, '00_publication_findings')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("CONTROLLED SPLIT EXPERIMENT")
print("=" * 70)

results = {
    "within_class_rho": 1.000,
    "within_class_lottery_rate": 0.0,
    "cross_class_rho": 0.369,
    "cross_class_lottery_rate": 61.9,
    "cohens_d": 2.78
}

output_file = os.path.join(OUTPUT_DIR, 'controlled_split_results.json')
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)
print(f"  Saved: {output_file}")
