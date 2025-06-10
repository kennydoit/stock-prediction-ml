import pickle
import pandas as pd
from pathlib import Path

# Path to model package
model_path = Path('data/model_outputs/enhanced_linear_regression_CRM_with_peers_20250610_080405.pkl')

with open(model_path, 'rb') as f:
    model_package = pickle.load(f)

coefs = model_package.get('coefficients', {})

print("Model package coefficients:")
for k, v in coefs.items():
    print(f"{k}: {v}")

# Optionally, compare to OLS audit CSV
audit_path = Path('data/audit/train_enhanced_model__CRM_ols_coefficients.csv')
if audit_path.exists():
    ols_df = pd.read_csv(audit_path)
    print("\nOLS audit coefficients:")
    for _, row in ols_df.iterrows():
        print(f"{row['feature']}: {row['coefficient']}")
    # Check for exact match
    ols_dict = dict(zip(ols_df['feature'], ols_df['coefficient']))
    mismatches = []
    for k, v in ols_dict.items():
        v2 = coefs.get(k, None)
        if v2 is None or abs(float(v) - float(v2)) > 1e-10:
            mismatches.append((k, v, v2))
    if not mismatches:
        print("\n✅ All coefficients match exactly!")
    else:
        print("\n❌ Mismatches found:")
        for k, v, v2 in mismatches:
            print(f"{k}: OLS={v}, Model={v2}")
else:
    print("No OLS audit CSV found.")
