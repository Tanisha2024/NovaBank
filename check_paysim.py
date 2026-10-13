
from pathlib import Path
import pandas as pd

file_path = Path("data/raw/paysim.csv")

if not file_path.exists():
    raise FileNotFoundError(f"Dataset not found: {file_path.resolve()}")

# Read only the header and a small sample first.
sample = pd.read_csv(file_path, nrows=5)

print("File:", file_path.resolve())
print("Columns:", sample.columns.tolist())
print("\nFirst 5 rows:")
print(sample)

# Count records without loading the entire dataset into memory.
with file_path.open("rb") as file:
    total_lines = sum(1 for _ in file) - 1

print(f"\nTotal transactions: {total_lines:,}")

# Count fraud labels in chunks.
fraud_count = 0
normal_count = 0

for chunk in pd.read_csv(file_path, chunksize=200_000):
    labels = pd.to_numeric(chunk["isFraud"], errors="coerce")
    fraud_count += int((labels == 1).sum())
    normal_count += int((labels == 0).sum())

print(f"Normal transactions: {normal_count:,}")
print(f"Fraud transactions: {fraud_count:,}")
print(f"Fraud rate: {fraud_count / total_lines * 100:.4f}%")