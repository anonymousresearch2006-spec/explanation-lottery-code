"""Fix all optimised_001 scripts that fail on Windows."""
import os

ENCODING_HEADER = (
    "import sys, io\n"
    "sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')\n"
    "sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')\n"
)

# Fix all optimised_001 scripts
scripts = sorted([f for f in os.listdir('.') if '_optimised_001_' in f and f.endswith('.py')])

for script in scripts:
    with open(script, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if 'TextIOWrapper' in content:
        print(f'  already patched: {script}')
        continue
    
    # Find end of module docstring
    first_triple = content.find('"""')
    if first_triple >= 0:
        second_triple = content.find('"""', first_triple + 3)
        if second_triple >= 0:
            nl = content.index('\n', second_triple)
            content = content[:nl+1] + '\n' + ENCODING_HEADER + '\n' + content[nl+1:]
    
    with open(script, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'  patched: {script}')

# Fix script 16 specifically: 'n' variable shadowing
with open('16_optimised_001_compas_legal.py', 'r', encoding='utf-8') as f:
    content = f.read()

# The issue: for n in preds shadows n = 3000
content = content.replace(
    "agree = np.all([preds[n] == preds['rf'] for n in preds], axis=0)",
    "agree = np.all([preds[nm] == preds['rf'] for nm in preds], axis=0)"
)

with open('16_optimised_001_compas_legal.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('  fixed variable shadowing in 16_optimised_001_compas_legal.py')

print('\nAll fixes applied!')
