"""
Script to generate exploratory data analysis visualizations for Heart Disease dataset inspection.
Outputs figures to reports/figures/.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_eda_figures(csv_path: str = "data/raw/heart.csv", output_dir: str = "reports/figures"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(csv_path)
    df_dedup = df_raw.drop_duplicates()

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # 1. Target Distribution Comparison (Raw vs Deduplicated)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    raw_counts = df_raw['target'].value_counts().sort_index()
    raw_labels = ['0: No Disease', '1: Disease']
    colors = ['#4A90E2', '#E74C3C']

    axes[0].bar(raw_labels, raw_counts.values, color=colors, edgecolor='black', alpha=0.85)
    axes[0].set_title(f"Raw Dataset Target (N={len(df_raw)})", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Count")
    for i, v in enumerate(raw_counts.values):
        pct = (v / len(df_raw)) * 100
        axes[0].text(i, v + 8, f"{v} ({pct:.1f}%)", ha='center', fontweight='bold')
    axes[0].set_ylim(0, max(raw_counts.values) * 1.15)

    dedup_counts = df_dedup['target'].value_counts().sort_index()
    axes[1].bar(raw_labels, dedup_counts.values, color=colors, edgecolor='black', alpha=0.85)
    axes[1].set_title(f"Deduplicated Target (N={len(df_dedup)})", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Count")
    for i, v in enumerate(dedup_counts.values):
        pct = (v / len(df_dedup)) * 100
        axes[1].text(i, v + 3, f"{v} ({pct:.1f}%)", ha='center', fontweight='bold')
    axes[1].set_ylim(0, max(dedup_counts.values) * 1.15)

    plt.suptitle("Heart Disease Dataset: Target Class Distribution", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(out_path / "target_distribution.png", dpi=200, bbox_inches='tight')
    plt.close()

    # 2. Duplicate Frequency Distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rep_counts = df_raw.value_counts().value_counts().sort_index()
    x_labels = [f"{k} occurrences" for k in rep_counts.index]
    bars = ax.bar(x_labels, rep_counts.values, color='#2ECC71', edgecolor='black', alpha=0.85)
    ax.set_title("Frequency of Exact Row Repetitions (Data Multiplication)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Number of Times a Patient Record is Repeated")
    ax.set_ylabel("Unique Patient Records Count")
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height} patients",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    ax.set_ylim(0, max(rep_counts.values) * 1.15)
    plt.tight_layout()
    fig.savefig(out_path / "duplicate_analysis.png", dpi=200, bbox_inches='tight')
    plt.close()

    # 3. Correlation with Target
    fig, ax = plt.subplots(figsize=(9, 5))
    corrs = df_raw.corr()['target'].drop('target').sort_values()
    bar_colors = ['#E74C3C' if c < 0 else '#3498DB' for c in corrs.values]
    y_pos = np.arange(len(corrs))
    ax.barh(y_pos, corrs.values, color=bar_colors, edgecolor='black', alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(corrs.index, fontweight='bold')
    ax.set_xlabel("Pearson Correlation Coefficient with Target")
    ax.set_title("Feature Correlation with Heart Disease Presence (Target)", fontsize=13, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    for i, v in enumerate(corrs.values):
        offset = 0.02 if v >= 0 else -0.06
        ax.text(v + offset, i, f"{v:.2f}", va='center', fontsize=9, fontweight='bold')
    ax.set_xlim(-0.55, 0.55)
    plt.tight_layout()
    fig.savefig(out_path / "correlation_with_target.png", dpi=200, bbox_inches='tight')
    plt.close()

    # 4. Numerical Distributions
    num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        ax = axes[i]
        ax.hist(df_raw[df_raw['target'] == 0][col], bins=20, alpha=0.6, label='No Disease (0)', color='#4A90E2', density=True)
        ax.hist(df_raw[df_raw['target'] == 1][col], bins=20, alpha=0.6, label='Disease (1)', color='#E74C3C', density=True)
        ax.set_title(f"Distribution of {col}", fontsize=11, fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    # Hide extra unused subplot
    axes[5].axis('off')
    # Add summary text box in the unused subplot
    axes[5].text(0.1, 0.5,
                "Key Observations:\n"
                "• Age & Trestbps higher in diseased patients\n"
                "• Thalach (max HR) noticeably lower in disease\n"
                "• Oldpeak shows strong right skew & ST depression\n"
                "• Chol has an extreme outlier (564 mg/dl)",
                fontsize=10, family='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F9FA", edgecolor="#BDC3C7"))

    plt.suptitle("Continuous Feature Distributions Stratified by Target", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.savefig(out_path / "numerical_distributions.png", dpi=200, bbox_inches='tight')
    plt.close()

    print(f"All EDA visualizations successfully generated in {out_path.resolve()}")


if __name__ == "__main__":
    generate_eda_figures()
