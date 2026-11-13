"""
Heart Disease Risk Assessment Agent - Dataset Inspection Script
Analyzes data/raw/heart.csv for schema, distributions, data quality, duplicates, and anomalies.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd


def inspect_dataset(csv_path: str = "data/raw/heart.csv") -> dict:
    file_path = Path(csv_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at {file_path.resolve()}")

    df = pd.read_csv(file_path)

    n_rows, n_cols = df.shape

    # Data types and missing values
    col_summary = []
    for col in df.columns:
        col_series = df[col]
        missing_count = int(col_series.isnull().sum())
        missing_pct = float((missing_count / n_rows) * 100)
        unique_vals = col_series.unique().tolist()
        n_unique = int(col_series.nunique())

        col_summary.append({
            "column": col,
            "dtype": str(col_series.dtype),
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "n_unique": n_unique,
            "unique_sample": unique_vals[:10] if n_unique > 10 else unique_vals,
        })

    # Duplicates
    dup_mask = df.duplicated()
    num_duplicates = int(dup_mask.sum())
    dup_pct = float((num_duplicates / n_rows) * 100)

    # Unique patient records (deduplicated)
    df_dedup = df.drop_duplicates()
    n_unique_rows = len(df_dedup)

    # Target analysis
    target_col = "target" if "target" in df.columns else df.columns[-1]
    target_counts = df[target_col].value_counts().to_dict()
    target_pcts = (df[target_col].value_counts(normalize=True) * 100).round(2).to_dict()

    # Deduplicated target counts
    dedup_target_counts = df_dedup[target_col].value_counts().to_dict()
    dedup_target_pcts = (df_dedup[target_col].value_counts(normalize=True) * 100).round(2).to_dict()

    # Categorical vs Numerical segregation
    # Binary / discrete categorical columns: sex, cp, fbs, restecg, exang, slope, ca, thal, target
    # Continuous / numerical columns: age, trestbps, chol, thalach, oldpeak
    numerical_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    categorical_cols = [c for c in df.columns if c not in numerical_cols and c != target_col]

    # Numerical statistics
    num_stats = {}
    for col in numerical_cols:
        if col in df.columns:
            s = df[col]
            q1 = float(s.quantile(0.25))
            q3 = float(s.quantile(0.75))
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers_iqr = int(((s < lower_bound) | (s > upper_bound)).sum())

            mean_val = float(s.mean())
            std_val = float(s.std())
            z_scores = np.abs((s - mean_val) / std_val) if std_val > 0 else np.zeros_like(s)
            outliers_z3 = int((z_scores > 3).sum())

            num_stats[col] = {
                "mean": round(mean_val, 2),
                "std": round(std_val, 2),
                "min": float(s.min()),
                "25%": round(q1, 2),
                "50%": round(float(s.median()), 2),
                "75%": round(q3, 2),
                "max": float(s.max()),
                "skewness": round(float(s.skew()), 3),
                "iqr_lower": round(lower_bound, 2),
                "iqr_upper": round(upper_bound, 2),
                "iqr_outliers": outliers_iqr,
                "z3_outliers": outliers_z3,
            }

    # Categorical distributions & Domain check
    cat_stats = {}
    for col in categorical_cols:
        cat_stats[col] = {
            "value_counts": df[col].value_counts().to_dict(),
            "percentages": (df[col].value_counts(normalize=True) * 100).round(2).to_dict()
        }

    # Anomaly / Quality checks
    anomalies = {}
    # ca in standard medical protocol is 0-3; check if ca == 4 exists
    if "ca" in df.columns:
        anomalies["ca_eq_4_count"] = int((df["ca"] == 4).sum())
    # thal standard categories are 1, 2, 3 (or 3, 6, 7); check for 0
    if "thal" in df.columns:
        anomalies["thal_eq_0_count"] = int((df["thal"] == 0).sum())
    # resting bp <= 0 or extreme
    if "trestbps" in df.columns:
        anomalies["trestbps_zero_or_neg"] = int((df["trestbps"] <= 0).sum())
        anomalies["trestbps_gt_200"] = int((df["trestbps"] > 200).sum())
    # chol <= 0 or extreme
    if "chol" in df.columns:
        anomalies["chol_zero_or_neg"] = int((df["chol"] <= 0).sum())
        anomalies["chol_gt_400"] = int((df["chol"] > 400).sum())

    # Correlation with target
    correlations = df.corr()[target_col].drop(target_col).round(4).to_dict()
    sorted_corr = dict(sorted(correlations.items(), key=lambda item: abs(item[1]), reverse=True))

    inspection_results = {
        "dataset_path": str(file_path),
        "shape": {"rows": n_rows, "columns": n_cols},
        "columns_summary": col_summary,
        "duplicates": {
            "count": num_duplicates,
            "percentage": round(dup_pct, 2),
            "unique_rows": n_unique_rows,
        },
        "target": {
            "column_name": target_col,
            "raw_distribution": target_counts,
            "raw_percentages": target_pcts,
            "dedup_distribution": dedup_target_counts,
            "dedup_percentages": dedup_target_pcts,
        },
        "numerical_stats": num_stats,
        "categorical_stats": cat_stats,
        "anomalies_and_quality": anomalies,
        "correlation_with_target": sorted_corr,
    }

    return inspection_results


if __name__ == "__main__":
    results = inspect_dataset()
    print("=" * 60)
    print("HEART DISEASE DATASET INSPECTION SUMMARY")
    print("=" * 60)
    print(f"Shape: {results['shape']['rows']} rows, {results['shape']['columns']} columns")
    print(f"Duplicates: {results['duplicates']['count']} ({results['duplicates']['percentage']}%)")
    print(f"Unique records: {results['duplicates']['unique_rows']}")
    print(f"Target column: '{results['target']['column_name']}'")
    print(f"Raw Target distribution: {results['target']['raw_distribution']} ({results['target']['raw_percentages']})")
    print(f"Dedup Target distribution: {results['target']['dedup_distribution']} ({results['target']['dedup_percentages']})")
    print("\nNumerical Features Summary:")
    for k, v in results['numerical_stats'].items():
        print(f"  - {k}: range=[{v['min']}, {v['max']}], mean={v['mean']}, std={v['std']}, IQR outliers={v['iqr_outliers']}, Z>3 outliers={v['z3_outliers']}")
    print("\nCategorical Features Unique Distributions:")
    for k, v in results['categorical_stats'].items():
        print(f"  - {k}: {v['value_counts']}")
    print("\nQuality / Anomaly Flags:")
    for k, v in results['anomalies_and_quality'].items():
        print(f"  - {k}: {v}")
    print("\nTop Correlations with Target (|r| desc):")
    for k, v in list(results['correlation_with_target'].items())[:7]:
        print(f"  - {k}: {v}")

    # Save output json for reporting
    out_file = Path("reports/dataset_inspection_summary.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed inspection written to {out_file.resolve()}")
