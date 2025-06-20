import sys
import joblib
from pathlib import Path
import pprint

# Usage: python test_dump_model_package.py <path_to_pkl>
if len(sys.argv) < 2:
    print("Usage: python test_dump_model_package.py <path_to_pkl>")
    sys.exit(1)

pkl_path = Path(sys.argv[1])
if not pkl_path.exists():
    print(f"File not found: {pkl_path}")
    sys.exit(1)

model_package = joblib.load(pkl_path)

# Dump to audit directory
output_dir = Path(__file__).parent.parent / 'data' / 'audit'
output_dir.mkdir(parents=True, exist_ok=True)
dump_path = output_dir / f"model_package_dump_{pkl_path.stem}.txt"

with open(dump_path, 'w', encoding='utf-8') as f:
    for key, value in model_package.items():
        f.write(f'===== {key} =====\n')
        if hasattr(value, 'to_string'):
            f.write(value.to_string())
            f.write('\n')
        elif isinstance(value, dict):
            f.write(pprint.pformat(value, indent=2, width=120))
            f.write('\n')
        else:
            f.write(str(value))
            f.write('\n')
        f.write('\n')
print(f"Model package dumped to {dump_path}")
