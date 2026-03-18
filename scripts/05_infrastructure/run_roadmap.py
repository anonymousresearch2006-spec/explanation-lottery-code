import os
import sys
import subprocess
import glob

def run_roadmap():
    print("=" * 70)
    print("EXECUTING OPTIMISED 001 ROADMAP (01 to 24)")
    print("=" * 70)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    roadmap_dir = os.path.join(repo_root, "scripts", "04_official_roadmap")
    
    # Get all python scripts, sorted alphabetically so 01 to 24 run in order
    scripts = sorted(glob.glob(os.path.join(roadmap_dir, "*.py")))
    if not scripts:
        print(f"Error: No scripts found in {roadmap_dir}")
        sys.exit(1)
        
    for script in scripts:
        script_name = os.path.basename(script)
        if script_name.startswith("__"): continue
        print(f"\n>> Running: {script_name} ...")
        
        try:
            subprocess.run([sys.executable, script], cwd=repo_root, check=True)
            print(f"   [OK] {script_name}")
        except subprocess.CalledProcessError as e:
            print(f"   [FAILED] {script_name} with exit code {e.returncode}")
            sys.exit(e.returncode)

    print("\n=" * 70)
    print("ALL ROADMAP EXPERIMENTS EXECUTED SUCCESSFULLY")
    print("Results saved in results/00_publication_findings")
    print("=" * 70)

if __name__ == "__main__":
    run_roadmap()
