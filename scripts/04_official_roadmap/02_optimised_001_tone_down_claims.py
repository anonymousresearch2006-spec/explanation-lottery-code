"""
=============================================================================
02_OPTIMISED_001: TONE DOWN THEORY CLAIMS
=============================================================================
Tier A -- Item 2 | Impact: 5/5 | Effort: 3-4 hours

Goal: Scan existing scripts/results for overclaiming language and generate
a comprehensive replacement mapping. Prevents theory reviewer attacks.

Output: results/optimised_001/02_tone_down_claims/
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
OUTPUT_DIR = os.path.join(RESULTS_DIR, 'optimised_001', '02_tone_down_claims')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCAN_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 70)
print("OPTIMISED_001 -- EXPERIMENT 02: TONE DOWN THEORY CLAIMS")
print("=" * 70)

# =============================================================================
# OVERCLAIM PATTERNS
# =============================================================================

# Words/phrases that are overclaiming in empirical papers
OVERCLAIM_PATTERNS = {
    'critical': {
        'prove': 'show / provide evidence',
        'proven': 'shown / supported by evidence',
        'proof': 'evidence / analysis',
        'theorem': 'observation / proposition',
        'theorems': 'observations / propositions',
        'guarantee': 'suggest / indicate',
        'guarantees': 'suggests / indicates',
        'establish': 'suggest / provide evidence for',
        'establishes': 'suggests / provides evidence for',
        'demonstrate conclusively': 'provide strong evidence',
        'definitively': 'strongly',
        'irrefutably': '(remove entirely)',
        'undeniably': 'notably',
        'always': 'typically / generally',
        'never': 'rarely / in our experiments',
        'impossible': 'very difficult / unlikely',
    },
    'moderate': {
        'demonstrate': 'show / observe',
        'verify': 'examine / check',
        'confirm': 'find evidence consistent with',
        'theoretical characterization': 'formal analysis / empirical characterization',
        'theoretical framework': 'analytical framework / empirical framework',
        'fundamental': 'important / notable',
        'crucial': 'important',
        'critical': 'notable / important',
        'novel discovery': 'novel finding / observation',
        'first to': 'among the first to / to our knowledge',
        'ground-breaking': 'notable',
        'significant contribution': 'contribution',
        'clearly shows': 'suggests / indicates',
        'proves that': 'provides evidence that',
    },
    'minor': {
        'ensure': 'aim to / seek to',
        'must': 'should / could',
        'will': 'may / could',
        'certainly': 'likely / plausibly',
        'obviously': '(remove or rephrase)',
        'clearly': 'our analysis suggests',
        'undoubtedly': 'likely',
        'no doubt': 'strong evidence suggests',
    }
}

# =============================================================================
# SCAN ALL PROJECT FILES
# =============================================================================
print("\n" + "=" * 70)
print("SCANNING PROJECT FILES FOR OVERCLAIMING LANGUAGE")
print("=" * 70)

found_claims = {'critical': [], 'moderate': [], 'minor': []}
file_stats = {}

py_files = [f for f in os.listdir(SCAN_DIR) if f.endswith('.py') and not f.startswith('__')]
md_files = [f for f in os.listdir(SCAN_DIR) if f.endswith('.md')]
scan_files = py_files + md_files

print(f"\nScanning {len(scan_files)} files...")

for filename in sorted(scan_files):
    filepath = os.path.join(SCAN_DIR, filename)
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            lines = content.split('\n')
    except Exception:
        continue

    file_claims = {'critical': 0, 'moderate': 0, 'minor': 0}

    for severity, patterns in OVERCLAIM_PATTERNS.items():
        for pattern, replacement in patterns.items():
            for line_num, line in enumerate(lines, 1):
                if re.search(r'\b' + re.escape(pattern) + r'\b', line, re.IGNORECASE):
                    found_claims[severity].append({
                        'file': filename,
                        'line': line_num,
                        'content': line.strip()[:120].encode('ascii', 'replace').decode(),
                        'pattern': pattern,
                        'replacement': replacement
                    })
                    file_claims[severity] += 1

    total = sum(file_claims.values())
    if total > 0:
        file_stats[filename] = file_claims
        print(f"  {filename}: {total} claims (C:{file_claims['critical']}, M:{file_claims['moderate']}, m:{file_claims['minor']})")

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================
print("\n" + "=" * 70)
print("OVERCLAIM SUMMARY STATISTICS")
print("=" * 70)

total_critical = len(found_claims['critical'])
total_moderate = len(found_claims['moderate'])
total_minor = len(found_claims['minor'])
total_all = total_critical + total_moderate + total_minor

print(f"\n  CRITICAL overclaims: {total_critical}")
print(f"  MODERATE overclaims: {total_moderate}")
print(f"  MINOR overclaims:    {total_minor}")
print(f"  TOTAL:               {total_all}")

# Most common patterns
print("\n  Most common overclaiming patterns:")
all_patterns = [c['pattern'] for c in found_claims['critical']] + \
               [c['pattern'] for c in found_claims['moderate']] + \
               [c['pattern'] for c in found_claims['minor']]
for pattern, count in Counter(all_patterns).most_common(15):
    severity = 'C' if pattern in OVERCLAIM_PATTERNS['critical'] else \
               ('M' if pattern in OVERCLAIM_PATTERNS['moderate'] else 'm')
    print(f"    [{severity}] '{pattern}' -> found {count} times")

# =============================================================================
# REPLACEMENT TABLE  
# =============================================================================
print("\n" + "=" * 70)
print("REPLACEMENT MAPPING TABLE")
print("=" * 70)

print(f"\n  {'SEVERITY':<10} {'ORIGINAL':<30} {'REPLACEMENT':<40}")
print(f"  {'-'*10} {'-'*30} {'-'*40}")

for severity in ['critical', 'moderate', 'minor']:
    for original, replacement in OVERCLAIM_PATTERNS[severity].items():
        sev_label = severity.upper()[:4]
        print(f"  {sev_label:<10} {original:<30} {replacement:<40}")

# =============================================================================
# SAVE RESULTS
# =============================================================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

results = {
    'summary': {
        'critical_count': total_critical,
        'moderate_count': total_moderate,
        'minor_count': total_minor,
        'total_count': total_all,
        'files_scanned': len(scan_files),
        'files_with_claims': len(file_stats)
    },
    'replacement_mapping': OVERCLAIM_PATTERNS,
    'file_statistics': file_stats,
    'critical_findings': found_claims['critical'][:50],
    'moderate_findings': found_claims['moderate'][:50],
}

output_file = os.path.join(OUTPUT_DIR, '02_tone_down_results.json')
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4, default=str)
print(f"\n  Saved: {output_file}")

# Detailed report
report_file = os.path.join(OUTPUT_DIR, '02_tone_down_report.txt')
with open(report_file, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("TONE DOWN CLAIMS -- DETAILED REPORT\n")
    f.write("=" * 70 + "\n\n")
    
    f.write(f"Total overclaims found: {total_all}\n")
    f.write(f"  Critical: {total_critical}\n")
    f.write(f"  Moderate: {total_moderate}\n")
    f.write(f"  Minor:    {total_minor}\n\n")
    
    f.write("CRITICAL FINDINGS (must fix):\n")
    f.write("-" * 70 + "\n")
    for c in found_claims['critical'][:30]:
        f.write(f"  File: {c['file']} (line {c['line']})\n")
        f.write(f"  Found: '{c['pattern']}' -> Replace with: '{c['replacement']}'\n")
        f.write(f"  Context: {c['content']}\n\n")

    f.write("\nMODERATE FINDINGS (should fix):\n")
    f.write("-" * 70 + "\n")
    for c in found_claims['moderate'][:30]:
        f.write(f"  File: {c['file']} (line {c['line']})\n")
        f.write(f"  Found: '{c['pattern']}' -> Replace with: '{c['replacement']}'\n")
        f.write(f"  Context: {c['content']}\n\n")

print(f"  Saved: {report_file}")
print("\n" + "=" * 70)
print("EXPERIMENT 02 COMPLETE")
print("=" * 70)
