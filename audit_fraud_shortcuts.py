from pathlib import Path
import json
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "fraud_model_data.csv"
REPORT_DIR = PROJECT_ROOT / "reports" / "fraud_detection" / "modeling"
FEATURE_AUDIT_FILE = REPORT_DIR / "fraud_feature_label_audit.csv"
RULE_METRICS_FILE = REPORT_DIR / "simple_balance_rule_metrics.csv"

CHUNK_SIZE = 200_000
REQUIRED = ["split", "type", "isFraud", "origin_balance_covers_amount"]


def safe_rate(numerator, denominator):
    return round(100 * numerator / denominator, 6) if denominator else 0.0


def main():
    if not DATA_FILE.is_file():
        raise FileNotFoundError(
            f"Prepared fraud data not found: {DATA_FILE}\n"
            "Run prepare_fraud_data.py first."
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Aggregate counts incrementally so the full 2.77M-row file need not be
    # held in memory just for this audit.
    feature_counts = {}
    type_counts = {}
    total_counts = {}

    for chunk in pd.read_csv(
        DATA_FILE,
        usecols=REQUIRED,
        dtype={
            "split": "category",
            "type": "category",
            "isFraud": "int8",
            "origin_balance_covers_amount": "int8",
        },
        chunksize=CHUNK_SIZE,
    ):
        chunk = chunk.dropna(subset=REQUIRED)
        for row in chunk.groupby(
            ["split", "isFraud", "origin_balance_covers_amount"], observed=True
        ).size().items():
            key = tuple(str(value) for value in row[0])
            feature_counts[key] = feature_counts.get(key, 0) + int(row[1])

        for row in chunk.groupby(["split", "type", "isFraud"], observed=True).size().items():
            key = tuple(str(value) for value in row[0])
            type_counts[key] = type_counts.get(key, 0) + int(row[1])

        # Simple diagnostic rule: predict fraud when origin balance does not
        # cover the transaction amount. This is only a shortcut diagnostic,
        # not proposed production logic.
        pred = chunk["origin_balance_covers_amount"].astype("int8").eq(0)
        actual = chunk["isFraud"].eq(1)
        for split_name, mask in chunk.groupby("split", observed=True).groups.items():
            local_index = chunk.index.isin(mask)
            y = actual[local_index]
            p = pred[local_index]
            tp = int((y & p).sum())
            fp = int((~y & p).sum())
            fn = int((y & ~p).sum())
            tn = int((~y & ~p).sum())
            key = str(split_name)
            current = total_counts.setdefault(key, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
            current["tp"] += tp
            current["fp"] += fp
            current["fn"] += fn
            current["tn"] += tn

    # Feature / target cross-tab with within-label proportions.
    feature_rows = []
    label_totals = {}
    for (split, label, covers), count in feature_counts.items():
        key = (split, label)
        label_totals[key] = label_totals.get(key, 0) + count
    for (split, label, covers), count in sorted(feature_counts.items()):
        denom = label_totals[(split, label)]
        feature_rows.append({
            "split": split,
            "isFraud": int(label),
            "origin_balance_covers_amount": int(covers),
            "transaction_count": count,
            "share_within_split_label_pct": safe_rate(count, denom),
        })
    feature_df = pd.DataFrame(feature_rows)
    feature_df.to_csv(FEATURE_AUDIT_FILE, index=False)

    # Contingency table by type lets us verify that all labels are concentrated
    # in the transaction types selected for this model.
    type_rows = [
        {"split": split, "type": tx_type, "isFraud": int(label), "transaction_count": count}
        for (split, tx_type, label), count in sorted(type_counts.items())
    ]
    type_df = pd.DataFrame(type_rows)

    rule_rows = []
    for split, counts in sorted(total_counts.items()):
        tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rule_rows.append({
            "split": split,
            "rule": "predict fraud if origin_balance_covers_amount == 0",
            "transactions": tp + fp + fn + tn,
            "fraud_cases": tp + fn,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "precision_pct": round(100 * precision, 6),
            "recall_pct": round(100 * recall, 6),
            "f1": round(f1, 6),
        })
    rule_df = pd.DataFrame(rule_rows)
    rule_df.to_csv(RULE_METRICS_FILE, index=False)

    print("Fraud shortcut / label audit finished.")
    print(f"Feature-label audit: {FEATURE_AUDIT_FILE}")
    print(f"Simple balance rule metrics: {RULE_METRICS_FILE}")
    print("\nOrigin balance covers amount, split by fraud label:")
    print(feature_df.to_string(index=False))
    print("\nSimple rule performance (diagnostic only):")
    print(rule_df.to_string(index=False))
    print("\nTransaction type / label counts:")
    print(type_df.to_string(index=False))
    print("\nInterpretation: a strong simple-rule score or extreme label association indicates the model may be learning PaySim simulation artefacts. These results are for leakage/shortcut investigation, not proof of real-world fraud detection performance.")


if __name__ == "__main__":
    main()
