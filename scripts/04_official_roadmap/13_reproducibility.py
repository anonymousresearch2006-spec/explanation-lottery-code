"""
=============================================================================
13_OPTIMISED_001: REPRODUCIBILITY PACKAGE
=============================================================================
Tier B -- Item 13 | Impact: 3/5 | Effort: 1 day

Goal: Create a reproducibility verification and documentation package.
TMLR values reproducibility highly.

Output: results/optimised_001/13_reproducibility/
=============================================================================
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


import sys
import os
import json
import importlib
import hashlib
import platform
from datetime import datetime

# Setup
PROJECT_DIR = 'results'
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
OUTPUT_DIR = os.path.join(RESULTS_DIR, 'optimised_001', '13_reproducibility')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCAN_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 70)
print("OPTIMISED_001 -- EXPERIMENT 13: REPRODUCIBILITY PACKAGE")
print("=" * 70)

# =============================================================================
# 1. ENVIRONMENT SPECIFICATION
# =============================================================================
print("\n" + "=" * 70)
print("1. ENVIRONMENT SPECIFICATION")
print("=" * 70)

env_info = {
    'python_version': sys.version,
    'platform': platform.platform(),
    'architecture': platform.architecture()[0],
    'processor': platform.processor(),
    'timestamp': datetime.now().isoformat()
}

print(f"\n  Python: {sys.version}")
print(f"  Platform: {platform.platform()}")

# Check required packages
REQUIRED_PACKAGES = {
    'numpy': 'numpy',
    'pandas': 'pandas',
    'scipy': 'scipy',
    'sklearn': 'scikit-learn',
    'xgboost': 'xgboost',
    'lightgbm': 'lightgbm',
    'catboost': 'catboost',
    'shap': 'shap',
    'matplotlib': 'matplotlib'
}

package_versions = {}
print(f"\n  Package versions:")
for import_name, pip_name in REQUIRED_PACKAGES.items():
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, '__version__', 'unknown')
        package_versions[pip_name] = version
        print(f"    {pip_name}: {version}")
    except ImportError:
        package_versions[pip_name] = 'NOT INSTALLED'
        print(f"    {pip_name}: NOT INSTALLED [!WARN!]")

env_info['packages'] = package_versions

# =============================================================================
# 2. REQUIREMENTS.TXT GENERATION
# =============================================================================
print("\n" + "=" * 70)
print("2. REQUIREMENTS.TXT")
print("=" * 70)

req_file = os.path.join(OUTPUT_DIR, 'requirements.txt')
with open(req_file, 'w') as f:
    for pip_name, version in package_versions.items():
        if version != 'NOT INSTALLED' and version != 'unknown':
            f.write(f"{pip_name}=={version}\n")
        elif version != 'NOT INSTALLED':
            f.write(f"{pip_name}\n")
print(f"\n  Generated: {req_file}")

# =============================================================================
# 3. DATA FILE CHECKSUMS
# =============================================================================
print("\n" + "=" * 70)
print("3. DATA FILE CHECKSUMS")
print("=" * 70)

data_files = {}
key_files = [
    os.path.join(RESULTS_DIR, 'combined_results.csv'),
    os.path.join(SCAN_DIR, 'elite_results.json'),
]

for filepath in key_files:
    if os.path.exists(filepath):
        file_size = os.path.getsize(filepath)
        # MD5 checksum
        with open(filepath, 'rb') as f:
            md5 = hashlib.md5(f.read()).hexdigest()
        data_files[os.path.basename(filepath)] = {
            'path': filepath,
            'size_bytes': file_size,
            'md5': md5
        }
        print(f"  {os.path.basename(filepath)}: {file_size:,} bytes, md5={md5[:12]}...")
    else:
        print(f"  {os.path.basename(filepath)}: NOT FOUND [!WARN!]")

# =============================================================================
# 4. CODE FILE INVENTORY
# =============================================================================
print("\n" + "=" * 70)
print("4. CODE FILE INVENTORY")
print("=" * 70)

code_files = {}
py_files = sorted([f for f in os.listdir(SCAN_DIR) if f.endswith('.py')])
print(f"\n  Total Python files: {len(py_files)}")

for filename in py_files:
    filepath = os.path.join(SCAN_DIR, filename)
    file_size = os.path.getsize(filepath)
    with open(filepath, 'rb') as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    code_files[filename] = {
        'size_bytes': file_size,
        'md5': md5
    }
    print(f"  {filename}: {file_size:,} bytes")

# =============================================================================
# 5. RESULT FILES INVENTORY
# =============================================================================
print("\n" + "=" * 70)
print("5. RESULT FILES INVENTORY")
print("=" * 70)

result_files = {}
for root, dirs, files in os.walk(RESULTS_DIR):
    for f in files:
        rel_path = os.path.relpath(os.path.join(root, f), RESULTS_DIR)
        file_size = os.path.getsize(os.path.join(root, f))
        result_files[rel_path] = file_size

print(f"\n  Total result files: {len(result_files)}")
print(f"  Total result size: {sum(result_files.values()):,} bytes")

# =============================================================================
# 6. EXECUTION INSTRUCTIONS
# =============================================================================
print("\n" + "=" * 70)
print("6. EXECUTION INSTRUCTIONS")
print("=" * 70)

instructions = """
REPRODUCTION INSTRUCTIONS
=========================

Prerequisites:
  1. Python 3.8+ installed
  2. Install dependencies: pip install -r requirements.txt

Data Preparation:
  1. Results are in results/combined_results.csv
  2. Original data is fetched from OpenML (automatic via API)

Running Experiments (in order):
  
  # Core experiment (generates combined_results.csv)
  python 01_session1.py
  python 02_session2.py
  
  # Optimised analysis suite
  python 01_optimised_001_novelty_surprise.py
  python 02_optimised_001_tone_down_claims.py
  python 03_optimised_001_design_choices.py
  python 04_optimised_001_synthetic_ground_truth.py
  python 05_optimised_001_dataset_level_analysis.py
  python 06_optimised_001_contribution_positioning.py
  python 07_optimised_001_reduce_overclaims.py
  python 08_optimised_001_related_work.py
  python 09_optimised_001_limitations.py
  python 10_optimised_001_mnist_experiment.py
  python 11_optimised_001_reliability_score.py
  python 12_optimised_001_figures_visualization.py
  python 13_optimised_001_reproducibility.py
  
  # Optional (Tier C)
  python 14_optimised_001_theoretical_deep.py
  python 15_optimised_001_extra_datasets.py
  python 16_optimised_001_compas_legal.py
  python 17_optimised_001_model_expansion.py

Expected Runtime:
  - Tier A (01-09): ~2-4 hours total
  - Tier B (10-13): ~2-6 hours total (MNIST is slowest)
  - Tier C (14-17): ~4-8 hours total

Random Seeds Used: 42, 123, 456 (for reproducibility)
"""

print(instructions)

inst_file = os.path.join(OUTPUT_DIR, 'REPRODUCTION_INSTRUCTIONS.txt')
with open(inst_file, 'w') as f:
    f.write(instructions)
print(f"  Saved: {inst_file}")

# =============================================================================
# 7. REPRODUCIBILITY VERIFICATION
# =============================================================================
print("\n" + "=" * 70)
print("7. REPRODUCIBILITY VERIFICATION")
print("=" * 70)

# Check if key results exist
verification = {
    'combined_results_exists': os.path.exists(os.path.join(RESULTS_DIR, 'combined_results.csv')),
    'elite_results_exists': os.path.exists(os.path.join(SCAN_DIR, 'elite_results.json')),
    'all_packages_available': all(v != 'NOT INSTALLED' for v in package_versions.values()),
    'figures_dir_exists': os.path.exists(os.path.join(PROJECT_DIR, 'figures')),
}

all_pass = all(verification.values())
for check, passed in verification.items():
    status = "[OK] PASS" if passed else "[FAIL] FAIL"
    print(f"  {status}: {check}")

print(f"\n  Overall: {'ALL CHECKS PASS [OK]' if all_pass else 'SOME CHECKS FAILED [FAIL]'}")

# =============================================================================
# SAVE MASTER REPRODUCIBILITY REPORT
# =============================================================================
report = {
    'environment': env_info,
    'data_files': data_files,
    'code_files': code_files,
    'result_file_count': len(result_files),
    'verification': verification,
    'all_checks_pass': all_pass
}

output_file = os.path.join(OUTPUT_DIR, '13_reproducibility_results.json')
with open(output_file, 'w') as f:
    json.dump(report, f, indent=4, default=str)
print(f"\n  Saved: {output_file}")

print("\n" + "=" * 70)
print("EXPERIMENT 13 COMPLETE")
print("=" * 70)
