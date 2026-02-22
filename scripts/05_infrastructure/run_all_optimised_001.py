"""
Runner script for all optimised_001 experiments.
Sets UTF-8 encoding to avoid Windows console issues, then runs all 17 experiments.
"""
import sys
import os
import io
import subprocess
import time

# Force UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

SCRIPTS = [
    '01_optimised_001_novelty_surprise.py',
    '02_optimised_001_tone_down_claims.py',
    '03_optimised_001_design_choices.py',
    '04_optimised_001_synthetic_ground_truth.py',
    '05_optimised_001_dataset_level_analysis.py',
    '06_optimised_001_contribution_positioning.py',
    '07_optimised_001_reduce_overclaims.py',
    '08_optimised_001_related_work.py',
    '09_optimised_001_limitations.py',
    '10_optimised_001_mnist_experiment.py',
    '11_optimised_001_reliability_score.py',
    '12_optimised_001_figures_visualization.py',
    '13_optimised_001_reproducibility.py',
    '14_optimised_001_theoretical_deep.py',
    '15_optimised_001_extra_datasets.py',
    '16_optimised_001_compas_legal.py',
    '17_optimised_001_model_expansion.py',
]

# Find project root (one level up from scripts/ infrastructure folder)
INFRA_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_ROOT = os.path.dirname(INFRA_DIR)
PROJECT_ROOT = os.path.dirname(SCRIPTS_ROOT)

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', '00_publication_findings')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Path to the roadmap folder
ROADMAP_DIR = os.path.join(SCRIPTS_ROOT, '04_official_roadmap')

print("=" * 70)
print(f"ROOT: {PROJECT_ROOT}")
print("OPTIMISED_001 — RUNNING ALL 17 EXPERIMENTS")
print("=" * 70)

env = os.environ.copy()
env['PYTHONIOENCODING'] = 'utf-8'

results_log = []
total_start = time.time()

for i, script in enumerate(SCRIPTS, 1):
    script_path = os.path.join(ROADMAP_DIR, script)
    if not os.path.exists(script_path):
        print(f"\n[{i:02d}/17] SKIP: {script} (not found at {script_path})")
        results_log.append({'script': script, 'status': 'SKIPPED', 'reason': 'not found'})
        continue
    
    print(f"\n{'='*70}")
    print(f"[{i:02d}/17] RUNNING: {script}")
    print(f"{'='*70}")
    
    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout per script
            encoding='utf-8',
            errors='replace',
            env=env
        )
        elapsed = time.time() - start
        
        if result.returncode == 0:
            print(f"  STATUS: SUCCESS ({elapsed:.1f}s)")
            # Print last 10 lines of output
            lines = result.stdout.strip().split('\n')
            for line in lines[-10:]:
                print(f"  {line}")
            results_log.append({'script': script, 'status': 'SUCCESS', 'time': f'{elapsed:.1f}s'})
        else:
            print(f"  STATUS: FAILED ({elapsed:.1f}s)")
            stderr_lines = result.stderr.strip().split('\n')
            for line in stderr_lines[-5:]:
                print(f"  ERROR: {line}")
            results_log.append({'script': script, 'status': 'FAILED', 'time': f'{elapsed:.1f}s', 
                              'error': stderr_lines[-1] if stderr_lines else 'unknown'})
    except subprocess.TimeoutExpired:
        print(f"  STATUS: TIMEOUT (>600s)")
        results_log.append({'script': script, 'status': 'TIMEOUT'})
    except Exception as e:
        print(f"  STATUS: ERROR ({e})")
        results_log.append({'script': script, 'status': 'ERROR', 'error': str(e)})

total_elapsed = time.time() - total_start

# Summary
print(f"\n\n{'='*70}")
print(f"EXECUTION SUMMARY")
print(f"{'='*70}")
print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} minutes)")

success = sum(1 for r in results_log if r['status'] == 'SUCCESS')
failed = sum(1 for r in results_log if r['status'] == 'FAILED')
errors = sum(1 for r in results_log if r['status'] in ('ERROR', 'TIMEOUT'))

print(f"  Success: {success}/17")
print(f"  Failed:  {failed}/17")
print(f"  Errors:  {errors}/17")

print(f"\n  {'Script':<50} {'Status':<10} {'Time'}")
print(f"  {'-'*50} {'-'*10} {'-'*10}")
for r in results_log:
    print(f"  {r['script']:<50} {r['status']:<10} {r.get('time', 'N/A')}")

# Save summary
import json
summary_file = os.path.join(RESULTS_DIR, 'execution_summary.json')
with open(summary_file, 'w', encoding='utf-8') as f:
    json.dump({
        'total_time_seconds': total_elapsed,
        'success_count': success,
        'failed_count': failed,
        'error_count': errors,
        'results': results_log
    }, f, indent=4)
print(f"\n  Summary saved: {summary_file}")

print(f"\n{'='*70}")
print(f"ALL EXPERIMENTS COMPLETE")
print(f"{'='*70}")
