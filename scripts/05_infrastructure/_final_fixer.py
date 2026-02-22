"""Final robust fixer for all optimised_001 scripts."""
import os
import re

REPLACEMENTS = {
    'ρ': 'rho',
    'τ': 'tau',
    '→': '->',
    'μ': 'mu',
    'σ': 'sigma',
    '★': '*',
    '✓': '[OK]',
    '✗': '[FAIL]',
    '±': '+/-',
    '—': '--',
    '⚠': '[!WARN!]',
    '≈': '~=',
    '≥': '>=',
    '≤': '<=',
}

def fix_content(content):
    # 1. Replace Unicode characters
    for k, v in REPLACEMENTS.items():
        content = content.replace(k, v)
    
    # 2. Fix np.isnan(rho) ambiguity
    # Find: if not np.isnan(rho):
    content = content.replace("if not np.isnan(rho):", "if not np.any(np.isnan(np.atleast_1d(rho))):")
    
    # 3. Fix SHAP shape issues (handle 3D arrays)
    # Target common SHAP processing blocks
    shap_fix = """                if isinstance(sv, list):
                    sv = sv[1]
                if len(getattr(sv, 'shape', [])) == 3:
                    sv = sv[:, :, 1]"""
    
    # Replace the existing SHAP list check
    content = content.replace(
        "                if isinstance(sv, list):\n                    sv = sv[1]",
        shap_fix
    )
    
    # 4. Fix specific script 16 issue if not already covered
    content = content.replace(
        "    agree = np.all([preds[nm] == preds['rf'] for nm in preds], axis=0)",
        "    agree = np.all([preds[nm] == preds[list(preds.keys())[0]] for nm in preds], axis=0)"
    )
    
    # 5. Fix any remaining truth value ambiguity in loops
    content = content.replace("for r in instance_rhos if not np.isnan(r)", "for r in instance_rhos if not np.any(np.isnan(np.atleast_1d(r)))")
    
    return content

# Process all roadmap scripts
scripts = sorted([f for f in os.listdir('.') if f.endswith('.py') and (f[:2].isdigit() and '_' in f)])

for script in scripts:
    print(f"Processing {script}...")
    with open(script, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = fix_content(content)
    
    with open(script, 'w', encoding='utf-8') as f:
        f.write(new_content)

print("\nAll scripts fixed and de-unicoded!")
