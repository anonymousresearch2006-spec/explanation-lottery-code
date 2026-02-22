"""
=============================================================================
07_OPTIMISED_001: REDUCE OVERCLAIMS (REGULATION / TRUST)
=============================================================================
Tier A -- Item 7 | Impact: 4/5 | Effort: 2 hours

Goal: Audit all regulation/trust claims and generate cautious replacements.
Prevents reviewer pushback on overblown practical claims.

Output: results/optimised_001/07_reduce_overclaims/
=============================================================================
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


import os
import re
import json
from collections import Counter

# Setup
PROJECT_DIR = 'results'
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
OUTPUT_DIR = os.path.join(RESULTS_DIR, 'optimised_001', '07_reduce_overclaims')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCAN_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 70)
print("OPTIMISED_001 -- EXPERIMENT 07: REDUCE OVERCLAIMS")
print("=" * 70)

# =============================================================================
# REGULATION / TRUST OVERCLAIM PATTERNS
# =============================================================================

REGULATION_CLAIMS = {
    'strong_regulation': [
        r'EU AI Act\s+(?:requires|mandates|demands)',
        r'(?:legally|regulatory)\s+(?:required|mandated|demanded)',
        r'compliance\s+(?:requires|mandates|demands)',
        r'(?:must|shall)\s+(?:comply|adhere|conform)',
        r'violat(?:es?|ing)\s+(?:EU|GDPR|AI Act)',
    ],
    'strong_trust': [
        r'cannot\s+be\s+trusted',
        r'should\s+(?:never|not)\s+(?:be\s+)?(?:trust|rely|use)',
        r'misleading\s+(?:explanation|feature|result)',
        r'dangerous(?:ly)?',
        r'harmful\s+(?:decision|outcome|explanation)',
    ],
    'strong_causation': [
        r'causes?\s+(?:harm|damage|bias|unfairness)',
        r'leads?\s+to\s+(?:harm|damage|bias)',
        r'results?\s+in\s+(?:harm|damage|discrimination)',
    ],
    'absolute_statements': [
        r'all\s+(?:explanations?|models?|predictions?)\s+(?:are|will)',
        r'no\s+(?:explanation|model|method)\s+(?:can|should)',
        r'every\s+(?:explanation|model|instance)',
    ]
}

REPLACEMENT_GUIDE = {
    'strong_regulation': {
        'before': 'EU AI Act requires transparent explanations',
        'after': 'The EU AI Act emphasizes the importance of transparent explainability, which our findings inform',
        'principle': 'Acknowledge regulation without claiming direct compliance implications'
    },
    'strong_trust': {
        'before': 'Explanations cannot be trusted',
        'after': 'Explanations should be interpreted with caution when cross-model agreement is low',
        'principle': 'Qualify trust claims with specific conditions'
    },
    'strong_causation': {
        'before': 'This causes harmful decisions',
        'after': 'This may contribute to suboptimal decision-making if not properly addressed',
        'principle': 'Use hedged language for causal claims'
    },
    'absolute_statements': {
        'before': 'All explanations are unreliable',
        'after': 'A significant proportion of explanations show low cross-model agreement',
        'principle': 'Replace absolutes with quantified proportions'
    }
}

# =============================================================================
# SCAN FILES
# =============================================================================
print("\n" + "=" * 70)
print("SCANNING FOR REGULATION / TRUST OVERCLAIMS")
print("=" * 70)

findings = {cat: [] for cat in REGULATION_CLAIMS}
files_scanned = 0

py_files = sorted([f for f in os.listdir(SCAN_DIR) if f.endswith('.py') and not f.startswith('__')])

for filename in py_files:
    filepath = os.path.join(SCAN_DIR, filename)
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        files_scanned += 1
    except Exception:
        continue

    for category, patterns in REGULATION_CLAIMS.items():
        for pattern in patterns:
            for line_num, line in enumerate(lines, 1):
                if re.search(pattern, line, re.IGNORECASE):
                    findings[category].append({
                        'file': filename,
                        'line': line_num,
                        'content': line.strip()[:150],
                        'pattern': pattern
                    })

# =============================================================================
# RESULTS
# =============================================================================
print(f"\n  Files scanned: {files_scanned}")
print(f"\n  FINDINGS BY CATEGORY:")
total_findings = 0
for category, items in findings.items():
    print(f"    {category}: {len(items)} instances")
    total_findings += len(items)
    for item in items[:5]:  # Show first 5
        print(f"      [{item['file']}:{item['line']}] {item['content'][:80]}")

print(f"\n  TOTAL OVERCLAIMS: {total_findings}")

# =============================================================================
# REPLACEMENT GUIDE
# =============================================================================
print("\n" + "=" * 70)
print("REPLACEMENT GUIDE")
print("=" * 70)

for category, guide in REPLACEMENT_GUIDE.items():
    print(f"\n  {category.upper()}:")
    print(f"    BEFORE: {guide['before']}")
    print(f"    AFTER:  {guide['after']}")
    print(f"    RULE:   {guide['principle']}")

# General cautious language patterns
CAUTIOUS_LANGUAGE = {
    'hedging_words': ['may', 'might', 'could', 'suggests', 'indicates', 'appears to'],
    'quantifying_words': ['approximately', 'in our experiments', 'under the conditions tested'],
    'qualifying_words': ['for the model families studied', 'on the datasets examined', 'within this experimental setup'],
    'action_softeners': ['consider', 'we recommend examining', 'practitioners may wish to', 'one approach is to']
}

print("\n" + "=" * 70)
print("CAUTIOUS LANGUAGE TOOLKIT")
print("=" * 70)

for category, words in CAUTIOUS_LANGUAGE.items():
    print(f"\n  {category}:")
    for w in words:
        print(f"    • {w}")

# =============================================================================
# SAVE RESULTS
# =============================================================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

results = {
    'scan_summary': {
        'files_scanned': files_scanned,
        'total_overclaims': total_findings,
        'by_category': {cat: len(items) for cat, items in findings.items()}
    },
    'findings': {cat: items[:20] for cat, items in findings.items()},
    'replacement_guide': REPLACEMENT_GUIDE,
    'cautious_language_toolkit': CAUTIOUS_LANGUAGE
}

output_file = os.path.join(OUTPUT_DIR, '07_overclaim_results.json')
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4, default=str)
print(f"\n  Saved: {output_file}")

print("\n" + "=" * 70)
print("EXPERIMENT 07 COMPLETE")
print("=" * 70)
