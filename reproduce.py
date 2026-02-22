import os
import sys
import subprocess
import time

BANNER = "=" * 70

def main():
    print(BANNER)
    print("EXPLANATION LOTTERY: REPRODUCIBILITY ENTRY POINT")
    print(BANNER)
    print("Run: python reproduce.py")

    # Ensure paths resolve correctly even if user runs from another folder
    repo_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_root)

    # 1. Check dependencies
    print("\n[STEP 1/3] Verifying core environment...")
    try:
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        import shap  # noqa: F401
        import sklearn  # noqa: F401
        print("  - Environment OK.")
        print(f"  - Python: {sys.version.split()[0]}")
        print(f"  - Working directory: {os.getcwd()}")
    except ImportError as e:
        print(f"  - ERROR: Missing dependency -> {e}")
        print("    Fix: pip install -r requirements.txt")
        return 2

    # 2. Data Readiness Check
    print("\n[STEP 2/3] Checking data readiness...")
    data_script = os.path.join("data", "download_data.py")
    cache_dir = os.path.join("data", "openml_cache")
    
    if not os.path.exists(cache_dir) or not os.listdir(cache_dir):
        print("  - Data cache missing or empty. Triggering automated download...")
        if os.path.exists(data_script):
            try:
                subprocess.run([sys.executable, data_script], check=True)
                print("  - Data preparation complete.")
            except subprocess.CalledProcessError:
                print("  - ERROR: Data download failed. Please run 'python data/download_data.py' manually.")
                return 1
        else:
            print(f"  - ERROR: Data script not found at {data_script}")
            return 1
    else:
        print("  - Data cache verified (OpenML assets present).")

    # 3. Run the official roadmap
    print("\n[STEP 3/3] Running the official experiment roadmap...")
    runner_path = os.path.join("scripts", "05_infrastructure", "run_roadmap.py")

    if not os.path.exists(runner_path):
        print(f"  - ERROR: Runner not found at: {runner_path}")
        print("    Check that the repository structure matches the README.")
        return 2

    start = time.time()
    try:
        # Propagate real-time logs to console
        subprocess.run([sys.executable, runner_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n  - ERROR: Roadmap failed (exit code {e.returncode}).")
        print("    Tip: Scroll up to see the first stack trace / error message.")
        return e.returncode or 1
    except KeyboardInterrupt:
        print("\n  - Interrupted by user.")
        return 130
    except Exception as e:
        print(f"\n  - ERROR during execution: {type(e).__name__}: {e}")
        return 1
    finally:
        elapsed = time.time() - start

    print("\n" + BANNER)
    print("REPRODUCTION COMPLETE.")
    print(f"Elapsed time: {elapsed/60:.1f} minutes")
    print("Results directory (primary): results/00_publication_findings/")
    print("Additional outputs may appear under: results/")
    print(BANNER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
