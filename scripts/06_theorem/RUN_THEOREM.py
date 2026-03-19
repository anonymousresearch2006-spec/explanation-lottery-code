"""
=============================================================================
THEOREM VALIDATION — MASTER RUNNER
=============================================================================
Runs all 3 theorem validation experiments in sequence:
  1. 01_compute_delta.py     — Compute Δ across all datasets
  2. 02_same_split_proof.py  — Same-split controlled experiment
  3. 03_dimensionality_effect.py — Dimensionality correlation

Produces: results/results/theorem_validation/theorem_summary.json

Usage:
  cd /path/to/explanation_lottery
  python scripts/06_theorem/RUN_THEOREM.py
=============================================================================
"""

import os
import sys
import subprocess
import time
import json

BANNER = "=" * 70

EXPERIMENTS = [
    ("01_compute_delta.py", "Compute Δ = ρ_intra − ρ_inter"),
    ("02_same_split_proof.py", "Same-Split Proof (key experiment)"),
    ("03_dimensionality_effect.py", "Dimensionality Effect (∂Δ/∂d > 0)"),
    (
        "04_three_way_comparison.py",
        "3-Way Hypothesis Boundary (Tree vs Linear vs Neural)",
    ),
]


def main():
    print(BANNER)
    print("THEOREM VALIDATION — MASTER RUNNER")
    print(BANNER)
    print(f"Experiments: {len(EXPERIMENTS)}")
    for i, (script, desc) in enumerate(EXPERIMENTS, 1):
        print(f"  {i}. {desc}")
    print(BANNER)

    # Ensure we run from repo root
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    os.chdir(repo_root)
    script_dir = os.path.join("scripts", "06_theorem")

    results = []
    total_start = time.time()

    for script_name, description in EXPERIMENTS:
        print(f"\n{'─'*70}")
        print(f"RUNNING: {description}")
        print(f"Script:  {script_name}")
        print(f"{'─'*70}")

        script_path = os.path.join(script_dir, script_name)
        if not os.path.exists(script_path):
            print(f"  ERROR: Script not found: {script_path}")
            results.append({"script": script_name, "status": "NOT_FOUND", "elapsed": 0})
            continue

        start = time.time()
        try:
            subprocess.run([sys.executable, script_path], check=True)
            elapsed = time.time() - start
            results.append(
                {
                    "script": script_name,
                    "description": description,
                    "status": "SUCCESS",
                    "elapsed_seconds": round(elapsed, 1),
                }
            )
            print(f"\n  ✓ {script_name} completed in {elapsed:.1f}s")
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start
            results.append(
                {
                    "script": script_name,
                    "description": description,
                    "status": f"FAILED (exit code {e.returncode})",
                    "elapsed_seconds": round(elapsed, 1),
                }
            )
            print(f"\n  ✗ {script_name} FAILED (exit code {e.returncode})")
        except Exception as e:
            elapsed = time.time() - start
            results.append(
                {
                    "script": script_name,
                    "description": description,
                    "status": f"ERROR: {str(e)[:100]}",
                    "elapsed_seconds": round(elapsed, 1),
                }
            )

    total_elapsed = time.time() - total_start

    # ── Load individual results for summary ───────────────────────────────
    output_dir = os.path.join("results", "results", "theorem_validation")
    verdicts = {}

    for fname, claim in [
        ("01_delta_results.json", "Δ > 0"),
        ("02_same_split_results.json", "Split-invariance"),
        ("03_dimensionality_results.json", "∂Δ/∂d > 0"),
        ("04_three_way_results.json", "Hypothesis Class Bound (3-Way)"),
    ]:
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                data = json.load(f)
            verdicts[claim] = data.get("verdict", "UNKNOWN")
        else:
            verdicts[claim] = "NOT_RUN"

    # ── Final Report ──────────────────────────────────────────────────────
    print("\n" + BANNER)
    print("THEOREM VALIDATION — FINAL REPORT")
    print(BANNER)

    all_confirmed = all(v == "CONFIRMED" for v in verdicts.values())

    for claim, verdict in verdicts.items():
        icon = "✓" if verdict == "CONFIRMED" else "~" if "PARTIAL" in verdict else "✗"
        print(f"  {icon} {claim:<25} : {verdict}")

    print(f"\n  Total runtime: {total_elapsed/60:.1f} minutes")

    if all_confirmed:
        print(f"\n  {'='*50}")
        print(f"  ALL THEOREM CLAIMS VALIDATED")
        print(f"  {'='*50}")
    else:
        print(f"\n  Some claims need review — see individual results.")

    # Save master summary
    master = {
        "title": "Theorem Validation Summary",
        "theorem": "Δ = ρ_intra - ρ_inter ≥ c > 0",
        "claims": verdicts,
        "experiments": results,
        "total_elapsed_seconds": round(total_elapsed, 1),
        "all_confirmed": all_confirmed,
    }

    summary_path = os.path.join(output_dir, "theorem_summary.json")
    with open(summary_path, "w") as f:
        json.dump(master, f, indent=2)
    print(f"\n  Saved: {summary_path}")

    print("\n" + BANNER)
    print("THEOREM VALIDATION COMPLETE")
    print(BANNER)

    return 0 if all_confirmed else 1


if __name__ == "__main__":
    raise SystemExit(main())
