"""
THE EXPLANATION LOTTERY - PROJECT SETUP
Run this FIRST to create proper folder structure
"""

import os
from datetime import datetime

PROJECT_NAME = "explanation_lottery"

folders = [
    PROJECT_NAME,
    f"{PROJECT_NAME}/data",
    f"{PROJECT_NAME}/results",
    f"{PROJECT_NAME}/results/session1",
    f"{PROJECT_NAME}/results/session2",
    f"{PROJECT_NAME}/checkpoints",
    f"{PROJECT_NAME}/figures",
    f"{PROJECT_NAME}/logs",
    f"{PROJECT_NAME}/paper",
]

print("="*60)
print("THE EXPLANATION LOTTERY - PROJECT SETUP")
print("="*60)

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"Created: {folder}/")

print("="*60)
print("SETUP COMPLETE!")
print("="*60)
print("")
print("Folder structure created:")
print("  explanation_lottery/")
print("    data/")
print("    results/session1/")
print("    results/session2/")
print("    checkpoints/")
print("    figures/")
print("    logs/")
print("    paper/")
print("")
print("NEXT: Run 01_session1.py")
