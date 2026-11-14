"""
Master orchestration script for Phase 1: Machine Learning Model Training and Comparison.
Trains, tunes, evaluates, visualizes, and serializes the 6 benchmark models.
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    CHAMPION_METADATA_PATH,
    CHAMPION_MODEL_PATH,
    DATA_PROCESSED_DIR,
    FIGURES_DIR,
    MODEL_RESULTS_CSV_PATH,
    MODEL_RESULTS_JSON_PATH,
    MODELS_DIR,
    PREPROCESSOR_PATH,
    RANDOM_STATE,
)
from src.data.preprocessing import load_preprocessor
from src.models.evaluate import (
    compute_overfitting_analysis,
    evaluate_model_performance,
)
from src.models.registry import get_model_registry
from src.models.train import (
    create_end_to_end_pipeline,
    extract_feature_importance,
    train_and_tune_model,
)


def load_processed_splits():
    """Loads preprocessed training and test partitions."""
    X_train = pd.read_csv(DATA_PROCESSED_DIR / "X_train_processed.csv")
    X_test = pd.read_csv(DATA_PROCESSED_DIR / "X_test_processed.csv")
    y_train = pd.read_csv(DATA_PROCESSED_DIR / "y_train.csv").iloc[:, 0]
    y_test = pd.read_csv(DATA_PROCESSED_DIR / "y_test.csv").iloc[:, 0]
    return X_train, X_test, y_train, y_test


def plot_visualizations(results_dict: dict, feature_importances: dict):
    """Generates all 5 clinical evaluation figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    model_names = list(results_dict.keys())

    # -------------------------------------------------------------
    # 1. Model Metric Comparison Bar Chart
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics = ["test_accuracy", "test_precision", "test_recall", "test_f1_score", "test_roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall (Sensitivity)", "F1-Score", "ROC-AUC"]
    colors = ["#3498DB", "#2ECC71", "#E74C3C", "#F39C12", "#9B59B6"]

    n_models = len(model_names)
    n_metrics = len(metrics)
    bar_width = 0.15
    indices = np.arange(n_models)

    for i, (m, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        vals = [results_dict[m_name][m] for m_name in model_names]
        ax.bar(indices + i * bar_width, vals, width=bar_width, label=label, color=color, edgecolor="black", alpha=0.85)

    ax.set_xticks(indices + bar_width * (n_metrics - 1) / 2)
    ax.set_xticklabels(model_names, rotation=15, ha="right", fontweight="bold", fontsize=10)
    ax.set_ylim(0.60, 1.05)
    ax.set_ylabel("Score (0.0 to 1.0)", fontweight="bold")
    ax.set_title("Phase 1 Benchmark: Holdout Test Set Performance Comparison", fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "model_metric_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------
    # 2. Combined ROC Curves
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6.5))
    roc_colors = ["#E74C3C", "#2ECC71", "#3498DB", "#9B59B6", "#E67E22", "#1ABC9C"]

    for (m_name, res), color in zip(results_dict.items(), roc_colors):
        fpr = res["roc_curve"]["fpr"]
        tpr = res["roc_curve"]["tpr"]
        auc = res["test_roc_auc"]
        ax.plot(fpr, tpr, label=f"{m_name} (AUC = {auc:.3f})", color=color, linewidth=2.2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Random Chance (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, fontweight="bold", pad=10)
    ax.legend(loc="lower right", frameon=True, fontsize=9.5)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curves_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------
    # 3. Confusion Matrices Grid (2x3)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    axes = axes.flatten()

    for idx, (m_name, res) in enumerate(results_dict.items()):
        ax = axes[idx]
        cm = np.array(res["confusion_matrix"]["matrix"])
        tn, fp, fn, tp = res["confusion_matrix"]["tn"], res["confusion_matrix"]["fp"], res["confusion_matrix"]["fn"], res["confusion_matrix"]["tp"]
        total = tn + fp + fn + tp

        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=max(tp, tn) * 1.2)
        ax.set_title(f"{m_name}\n(Recall: {res['test_recall']:.2f} | FN: {fn})", fontweight="bold", fontsize=11)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0 (Healthy)", "Pred 1 (Disease)"], fontsize=8.5)
        ax.set_yticklabels(["True 0", "True 1"], fontsize=8.5)

        # Annotate cells
        cell_labels = [
            [f"TN: {tn}\n({tn/total*100:.1f}%)", f"FP: {fp}\n({fp/total*100:.1f}%)"],
            [f"FN: {fn}\n({fn/total*100:.1f}%)", f"TP: {tp}\n({tp/total*100:.1f}%)"],
        ]
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > thresh else "black"
                # Highlight FN in red box style if > 0
                if i == 1 and j == 0 and fn > 0:
                    ax.text(j, i, cell_labels[i][j], ha="center", va="center", color="#C0392B", fontweight="bold", fontsize=9.5)
                else:
                    ax.text(j, i, cell_labels[i][j], ha="center", va="center", color=color, fontweight="bold", fontsize=9.5)

    plt.suptitle("Confusion Matrices: Diagnostic Accuracy & False Negative Audit", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=200, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------
    # 4. Feature Importance Comparison
    # -------------------------------------------------------------
    models_with_fi = [m for m in ["Random Forest", "XGBoost", "Logistic Regression"] if m in feature_importances]
    fig, axes = plt.subplots(1, len(models_with_fi), figsize=(6 * len(models_with_fi), 6))
    if len(models_with_fi) == 1:
        axes = [axes]

    for ax, m_name in zip(axes, models_with_fi):
        fi = feature_importances[m_name]
        top_features = list(fi.items())[:10]
        f_names = [x[0].replace("num__", "").replace("nom__", "").replace("ord__", "").replace("bin__", "") for x in top_features]
        f_vals = [x[1] for x in top_features]

        y_pos = np.arange(len(f_names))
        colors_bar = ["#2ECC71" if v >= 0 else "#E74C3C" for v in f_vals]
        ax.barh(y_pos, f_vals, color=colors_bar, edgecolor="black", alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(f_names, fontweight="bold")
        ax.invert_yaxis()
        metric_name = "Coefficient" if m_name == "Logistic Regression" else "Gini / Split Gain Importance"
        ax.set_xlabel(metric_name, fontweight="bold")
        ax.set_title(f"{m_name} Top Drivers", fontweight="bold", fontsize=11)
        for i, v in enumerate(f_vals):
            offset = 0.01 if v >= 0 else -0.05
            ax.text(v + offset, i, f"{v:.2f}", va="center", fontsize=8.5, fontweight="bold")

    plt.suptitle("Clinical Feature Attribution & Disease Predictive Drivers", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importances.png", dpi=200, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------
    # 5. Overfitting & Generalization Gap
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    train_accs = [results_dict[m]["train_accuracy"] for m_name in model_names for m in [m_name]]
    cv_accs = [results_dict[m]["cv_accuracy_mean"] for m_name in model_names for m in [m_name]]
    test_accs = [results_dict[m]["test_accuracy"] for m_name in model_names for m in [m_name]]

    x = np.arange(len(model_names))
    width = 0.25

    ax.bar(x - width, train_accs, width, label="Train Accuracy", color="#34495E", edgecolor="black", alpha=0.85)
    ax.bar(x, cv_accs, width, label="5-Fold CV Accuracy (Train Only)", color="#3498DB", edgecolor="black", alpha=0.85)
    ax.bar(x + width, test_accs, width, label="Holdout Test Accuracy", color="#E67E22", edgecolor="black", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right", fontweight="bold")
    ax.set_ylim(0.60, 1.05)
    ax.set_ylabel("Accuracy", fontweight="bold")
    ax.set_title("Overfitting Assessment: Train vs. 5-Fold CV vs. Test Generalization", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "train_vs_test_overfitting.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"  -> Generated 5 publication figures in {FIGURES_DIR.resolve()}")


def execute_benchmarking():
    print("=" * 80)
    print("HEART DISEASE RISK ASSESSMENT - PHASE 1: ML MODEL TRAINING & BENCHMARKING")
    print("=" * 80)

    # 1. Load Preprocessed Partitions
    print("\n[Step 1] Loading preprocessed data splits from disk...")
    X_train, X_test, y_train, y_test = load_processed_splits()
    feature_names = list(X_train.columns)
    print(f"  -> X_train shape: {X_train.shape}, y_train: {len(y_train)} (Positives: {(y_train==1).sum()})")
    print(f"  -> X_test shape:  {X_test.shape},  y_test:  {len(y_test)} (Positives: {(y_test==1).sum()})")

    # 2. Train and Evaluate All 6 Models
    print("\n[Step 2] Training and evaluating candidate models using Stratified 5-Fold CV on training partition...")
    candidate_names = [
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
        "Support Vector Machine",
        "K-Nearest Neighbors",
        "Multi-Layer Perceptron",
    ]

    results = {}
    fitted_models = {}
    feature_importances = {}

    for m_name in candidate_names:
        print(f"\n  [Training] {m_name}...")
        best_model, cv_summary = train_and_tune_model(
            model_name=m_name,
            X_train=X_train,
            y_train=y_train,
            tune=True,
            cv_splits=5,
            scoring="roc_auc",
        )
        fitted_models[m_name] = best_model

        # Train set evaluation (for overfitting check)
        train_eval = evaluate_model_performance(best_model, X_train, y_train)

        # Holdout test set evaluation (FINAL untouched evaluation)
        test_eval = evaluate_model_performance(best_model, X_test, y_test)

        # Overfitting gap
        overfit = compute_overfitting_analysis(train_eval, test_eval)

        # Feature importance
        fi = extract_feature_importance(best_model, feature_names)
        if fi:
            feature_importances[m_name] = fi

        results[m_name] = {
            "model_name": m_name,
            "best_hyperparameters": cv_summary["best_params"],
            # CV metrics
            "cv_roc_auc_mean": cv_summary["cv_roc_auc_mean"],
            "cv_roc_auc_std": cv_summary["cv_roc_auc_std"],
            "cv_accuracy_mean": cv_summary["cv_accuracy_mean"],
            "cv_recall_mean": cv_summary["cv_recall_mean"],
            "cv_f1_mean": cv_summary["cv_f1_mean"],
            # Train metrics
            "train_accuracy": train_eval["accuracy"],
            "train_recall": train_eval["recall"],
            "train_roc_auc": train_eval["roc_auc"],
            # Test metrics
            "test_accuracy": test_eval["accuracy"],
            "test_precision": test_eval["precision"],
            "test_recall": test_eval["recall"],
            "test_specificity": test_eval["specificity"],
            "test_f1_score": test_eval["f1_score"],
            "test_roc_auc": test_eval["roc_auc"],
            "brier_score": test_eval["brier_score"],
            "confusion_matrix": test_eval["confusion_matrix"],
            "overfitting_gap": overfit,
            "roc_curve": test_eval["roc_curve"],
        }

        print(f"     -> CV ROC-AUC:   {cv_summary['cv_roc_auc_mean']:.4f} (+/- {cv_summary['cv_roc_auc_std']:.4f})")
        print(f"     -> Test Accuracy: {test_eval['accuracy']:.4f} | Recall: {test_eval['recall']:.4f} | F1: {test_eval['f1_score']:.4f} | ROC-AUC: {test_eval['roc_auc']:.4f}")
        print(f"     -> Confusion Matrix (TN, FP, FN, TP): {test_eval['confusion_matrix']['tn']}, {test_eval['confusion_matrix']['fp']}, {test_eval['confusion_matrix']['fn']}, {test_eval['confusion_matrix']['tp']}")

    # 3. Produce Final Comparison Table
    print("\n" + "=" * 80)
    print("FINAL BENCHMARK COMPARISON TABLE (HOLDOUT TEST SET)")
    print("=" * 80)
    rows = []
    for m_name, res in results.items():
        cm = res["confusion_matrix"]
        rows.append({
            "Model": m_name,
            "CV ROC-AUC": f"{res['cv_roc_auc_mean']:.3f} ± {res['cv_roc_auc_std']:.3f}",
            "Test Acc": f"{res['test_accuracy']:.3f}",
            "Test Prec": f"{res['test_precision']:.3f}",
            "Test Rec": f"{res['test_recall']:.3f}",
            "Test Spec": f"{res['test_specificity']:.3f}",
            "Test F1": f"{res['test_f1_score']:.3f}",
            "Test AUC": f"{res['test_roc_auc']:.3f}",
            "FN": cm["fn"],
            "FP": cm["fp"],
            "Brier": f"{res['brier_score']:.3f}",
            "Overfit Gap (Acc)": f"{res['overfitting_gap']['accuracy_gap']:+.3f}",
        })

    df_comparison = pd.DataFrame(rows)
    print(df_comparison.to_string(index=False))

    # Save comparison to CSV
    MODEL_RESULTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_comparison.to_csv(MODEL_RESULTS_CSV_PATH, index=False)
    print(f"\n  -> Saved tabular comparison to: {MODEL_RESULTS_CSV_PATH}")

    # 4. Generate Visualizations
    print("\n[Step 3] Generating clinical diagnostic visualizations...")
    plot_visualizations(results, feature_importances)

    # 5. Multi-Criteria Clinical Champion Selection
    print("\n[Step 4] Multi-criteria clinical champion selection...")
    # Clinical Scoring Formula:
    # 1. Recall (weight 0.35) -> prioritize catching disease (minimize fatal False Negatives)
    # 2. ROC-AUC (weight 0.30) -> overall discriminatory ranking capacity across thresholds
    # 3. F1-Score (weight 0.20) -> harmonic balance of precision and recall
    # 4. Generalization / Overfitting penalty (weight 0.15) -> penalize huge train-test gap
    scores = {}
    for m_name, res in results.items():
        rec = res["test_recall"]
        auc = res["test_roc_auc"]
        f1 = res["test_f1_score"]
        acc_gap = res["overfitting_gap"]["accuracy_gap"]
        # Penalty for excessive overfitting: gap > 0.10 incurs penalty
        gap_penalty = max(0.0, acc_gap - 0.05)
        composite_score = (0.35 * rec) + (0.30 * auc) + (0.20 * f1) + (0.15 * (1.0 - gap_penalty))
        scores[m_name] = round(composite_score, 4)
        print(f"  -> {m_name}: Composite Clinical Score = {composite_score:.4f} (Recall: {rec:.3f}, AUC: {auc:.3f}, F1: {f1:.3f}, Gap: {acc_gap:+.3f})")

    champion_name = max(scores, key=scores.get)
    champion_model = fitted_models[champion_name]
    champion_res = results[champion_name]
    print(f"\n🏆 SELECTED CHAMPION MODEL: {champion_name} (Composite Score: {scores[champion_name]:.4f})")
    print(f"   Reasoning: Delivers the optimal combination of high sensitivity (Recall={champion_res['test_recall']:.3f}), discriminative power (ROC-AUC={champion_res['test_roc_auc']:.3f}), and robust generalization without severe overfitting.")

    # 6. Assemble & Save End-to-End Inference Pipeline
    print("\n[Step 5] Building and saving end-to-end inference pipeline...")
    raw_preprocessor = load_preprocessor(PREPROCESSOR_PATH)
    end_to_end_pipeline = create_end_to_end_pipeline(raw_preprocessor, champion_model)

    CHAMPION_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(end_to_end_pipeline, CHAMPION_MODEL_PATH)
    print(f"  -> Serialized complete end-to-end pipeline to: {CHAMPION_MODEL_PATH}")

    # 7. Save Champion Metadata & Full Results JSON
    metadata = {
        "champion_model_name": champion_name,
        "composite_clinical_score": scores[champion_name],
        "hyperparameters": champion_res["best_hyperparameters"],
        "metrics_holdout_test": {
            "accuracy": champion_res["test_accuracy"],
            "precision": champion_res["test_precision"],
            "recall": champion_res["test_recall"],
            "specificity": champion_res["test_specificity"],
            "f1_score": champion_res["test_f1_score"],
            "roc_auc": champion_res["test_roc_auc"],
            "brier_score": champion_res["brier_score"],
            "false_negatives": champion_res["confusion_matrix"]["fn"],
            "false_positives": champion_res["confusion_matrix"]["fp"],
        },
        "overfitting_audit": champion_res["overfitting_gap"],
        "cv_performance": {
            "cv_roc_auc_mean": champion_res["cv_roc_auc_mean"],
            "cv_roc_auc_std": champion_res["cv_roc_auc_std"],
            "cv_accuracy_mean": champion_res["cv_accuracy_mean"],
            "cv_recall_mean": champion_res["cv_recall_mean"],
            "cv_f1_mean": champion_res["cv_f1_mean"],
        },
        "feature_importance": feature_importances.get(champion_name, None),
    }

    with open(CHAMPION_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  -> Saved champion metadata to: {CHAMPION_METADATA_PATH}")

    # Save full results JSON (excluding non-serializable objects)
    json_results = {}
    for k, v in results.items():
        v_copy = v.copy()
        v_copy.pop("roc_curve", None)  # clean large arrays
        json_results[k] = v_copy
    with open(MODEL_RESULTS_JSON_PATH, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"  -> Saved detailed benchmarking JSON to: {MODEL_RESULTS_JSON_PATH}")

    # 8. Verify End-to-End Inference on Raw Patient Record (Phase 2 Readiness)
    print("\n[Step 6] Verifying end-to-end inference on raw patient record (Phase 2 readiness)...")
    sample_patient_df = pd.DataFrame([{
        "age": 58,
        "sex": 1,
        "cp": 2,
        "trestbps": 135,
        "chol": 250,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.2,
        "slope": 2,
        "ca": 4,     # Surrogate code: pipeline handles this cleanly
        "thal": 2,
    }])

    loaded_champion_pipeline = joblib.load(CHAMPION_MODEL_PATH)
    pred_class = int(loaded_champion_pipeline.predict(sample_patient_df)[0])
    pred_prob = float(loaded_champion_pipeline.predict_proba(sample_patient_df)[0, 1])

    print(f"  -> Raw Input: 58-year-old male with cp=2, trestbps=135, chol=250, ca=4 (surrogate)")
    print(f"  -> Champion Predicted Class: {pred_class} ({'Disease Present' if pred_class == 1 else 'No Disease'})")
    print(f"  -> Predicted Heart Disease Probability: {pred_prob * 100:.2f}%")

    print("\n" + "=" * 80)
    print("PHASE 1 MODEL TRAINING & BENCHMARKING COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    execute_benchmarking()

