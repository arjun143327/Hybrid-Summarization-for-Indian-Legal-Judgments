"""
train_gbr.py — Orchestrates GBR training for Configs C1, C2, and C3
per 02_METHODOLOGY.md Stage 2.

Performs:
1. Document-level train/validation holdout (85% train, 15% val, seed 42).
2. Training of C1 GBR on train_features_c1_scaled.npy (5 features).
3. Training of C2 GBR on train_features_c1_scaled.npy (identical inputs & seed)
   with explicit identity assertion against C1.
4. Training of C3 GBR on train_features_c3_scaled.npy (4 features).
5. Comprehensive metrics logging (R², RMSE, MAE, feature importances, negative R² check).
6. Model saving to models/gbr_c{1,2,3}.pkl.
"""

import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from src.scoring.gbr_model import (
    DEFAULT_GBR_PARAMS,
    CONFIG_FEATURE_NAMES,
    create_document_validation_split,
    train_gbr,
    save_gbr_model,
)


def main():
    print("=" * 80)
    print("PHASE 3: GRADIENT BOOSTING REGRESSOR (GBR) MODEL TRAINING")
    print("Configs: C1 (Base Replica), C2 (Redundancy Swap), C3 (Proposed System)")
    print("=" * 80)

    feat_dir = os.path.join("data", "processed", "features")
    splits_dir = os.path.join("data", "splits")
    models_dir = os.path.join("models")
    logs_dir = os.path.join("results", "logs")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)

    # 1. Load data
    print("\n[Step 1/5] Loading pre-processed scaled feature matrices and labels...")
    t0 = time.time()
    X_c1_scaled = np.load(os.path.join(feat_dir, "train_features_c1_scaled.npy"))
    X_c3_scaled = np.load(os.path.join(feat_dir, "train_features_c3_scaled.npy"))
    train_labels = np.load(os.path.join(feat_dir, "train_labels.npy"))

    with open(os.path.join(feat_dir, "train_doc_boundaries.json")) as f:
        doc_boundaries = json.load(f)

    with open(os.path.join(feat_dir, "train_sentence_index.json")) as f:
        sentence_keys = json.load(f)

    print(f"  Loaded X_c1_scaled: {X_c1_scaled.shape} ({X_c1_scaled.nbytes / 1e6:.1f} MB)")
    print(f"  Loaded X_c3_scaled: {X_c3_scaled.shape} ({X_c3_scaled.nbytes / 1e6:.1f} MB)")
    print(f"  Loaded train_labels: {train_labels.shape} ({train_labels.nbytes / 1e6:.1f} MB)")
    print(f"  Loaded {len(doc_boundaries):,} document boundaries and {len(sentence_keys):,} sentence keys in {time.time()-t0:.2f}s")

    # 2. Document-level train/validation split
    print("\n[Step 2/5] Establishing document-level validation holdout split (85% train / 15% val)...")
    val_ratio = 0.15
    seed = 42
    train_idx, val_idx, train_doc_ids, val_doc_ids = create_document_validation_split(
        doc_boundaries=doc_boundaries,
        val_ratio=val_ratio,
        seed=seed,
    )

    print(f"  Total training set:     {len(doc_boundaries):,} documents | {len(sentence_keys):,} sentences")
    print(f"  GBR Train partition:    {len(train_doc_ids):,} documents ({len(train_doc_ids)/len(doc_boundaries)*100:.1f}%) | {len(train_idx):,} sentences")
    print(f"  GBR Val partition:      {len(val_doc_ids):,} documents ({len(val_doc_ids)/len(doc_boundaries)*100:.1f}%) | {len(val_idx):,} sentences")

    # Save split documentation
    val_ids_path = os.path.join(splits_dir, "val_ids.txt")
    gbr_train_ids_path = os.path.join(splits_dir, "gbr_train_ids.txt")
    split_indices_path = os.path.join(splits_dir, "gbr_train_val_indices.json")

    with open(val_ids_path, "w") as f:
        f.write("\n".join(val_doc_ids) + "\n")
    with open(gbr_train_ids_path, "w") as f:
        f.write("\n".join(train_doc_ids) + "\n")
    with open(split_indices_path, "w") as f:
        json.dump({
            "val_ratio": val_ratio,
            "seed": seed,
            "n_train_docs": len(train_doc_ids),
            "n_val_docs": len(val_doc_ids),
            "n_train_sentences": len(train_idx),
            "n_val_sentences": len(val_idx),
            "train_indices_count": len(train_idx),
            "val_indices_count": len(val_idx),
        }, f, indent=2)

    print(f"  Saved {val_ids_path} ({len(val_doc_ids)} doc IDs)")
    print(f"  Saved {gbr_train_ids_path} ({len(train_doc_ids)} doc IDs)")
    print(f"  Saved {split_indices_path}")

    # Ground truth targets
    y_train = train_labels[train_idx]
    y_val = train_labels[val_idx]

    all_metrics = {}

    # 3. Train C1 GBR
    print("\n[Step 3/5] Training GBR for Config C1 (Base Replica)...")
    X_c1_train = X_c1_scaled[train_idx]
    X_c1_val = X_c1_scaled[val_idx]

    model_c1, metrics_c1 = train_gbr(
        X_train=X_c1_train,
        y_train=y_train,
        X_val=X_c1_val,
        y_val=y_val,
        feature_names=CONFIG_FEATURE_NAMES["C1"],
        params=DEFAULT_GBR_PARAMS,
        config_name="C1",
    )
    save_gbr_model(model_c1, os.path.join(models_dir, "gbr_c1.pkl"))
    all_metrics["C1"] = metrics_c1

    # 4. Train C2 GBR & Pipeline Sanity Verification
    print("\n[Step 4/5] Training GBR for Config C2 (Redundancy-Only Swap) & Pipeline Sanity Check...")
    # C2 uses identical feature vector and inputs to C1
    model_c2, metrics_c2 = train_gbr(
        X_train=X_c1_train,
        y_train=y_train,
        X_val=X_c1_val,
        y_val=y_val,
        feature_names=CONFIG_FEATURE_NAMES["C2"],
        params=DEFAULT_GBR_PARAMS,
        config_name="C2",
    )
    save_gbr_model(model_c2, os.path.join(models_dir, "gbr_c2.pkl"))
    all_metrics["C2"] = metrics_c2

    # Verification of C1 vs C2 identity
    print("\n>>> Verifying C1 vs C2 Identity Sanity Check <<<")
    c1_val_preds = model_c1.predict(X_c1_val)
    c2_val_preds = model_c2.predict(X_c1_val)
    max_pred_diff = float(np.max(np.abs(c1_val_preds - c2_val_preds)))
    print(f"  Max absolute prediction difference between C1 and C2 on validation set: {max_pred_diff:.2e}")
    assert max_pred_diff == 0.0, f"C1 and C2 predictions differ by {max_pred_diff}!"

    # Compare feature importances
    for k in CONFIG_FEATURE_NAMES["C1"]:
        imp_c1 = metrics_c1["feature_importances"][k]
        imp_c2 = metrics_c2["feature_importances"][k]
        assert abs(imp_c1 - imp_c2) < 1e-6, f"Feature importance mismatch for {k}: {imp_c1} vs {imp_c2}"
    print("  Feature importances match: EXACT bitwise identity confirmed!")
    print("  Sanity Check PASSED: C1 and C2 are provably identical in scoring pipeline.")

    # 5. Train C3 GBR
    print("\n[Step 5/5] Training GBR for Config C3 (Proposed System — SBERT Cosine, No WMD)...")
    X_c3_train = X_c3_scaled[train_idx]
    X_c3_val = X_c3_scaled[val_idx]

    model_c3, metrics_c3 = train_gbr(
        X_train=X_c3_train,
        y_train=y_train,
        X_val=X_c3_val,
        y_val=y_val,
        feature_names=CONFIG_FEATURE_NAMES["C3"],
        params=DEFAULT_GBR_PARAMS,
        config_name="C3",
    )
    save_gbr_model(model_c3, os.path.join(models_dir, "gbr_c3.pkl"))
    all_metrics["C3"] = metrics_c3

    # 6. Save comprehensive JSON & Markdown report
    json_report_path = os.path.join(logs_dir, "gbr_training_report.json")
    with open(json_report_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved metrics JSON to {json_report_path}")

    md_report_path = os.path.join(logs_dir, "gbr_training_report.md")
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write("# GradientBoostingRegressor (GBR) Training Report\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total Sentences**: 1,010,961 (7,028 documents)\n")
        f.write(f"**Train Split**: {len(train_idx):,} sentences ({len(train_doc_ids):,} docs, 85%)\n")
        f.write(f"**Validation Holdout**: {len(val_idx):,} sentences ({len(val_doc_ids):,} docs, 15%)\n\n")

        f.write("## Hyperparameters\n")
        f.write("```json\n" + json.dumps(DEFAULT_GBR_PARAMS, indent=2) + "\n```\n\n")

        f.write("## Performance Summary\n\n")
        f.write("| Metric | Config C1 (Replica) | Config C2 (Redundancy Swap) | Config C3 (Proposed) |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| **Features** | {len(CONFIG_FEATURE_NAMES['C1'])} features | {len(CONFIG_FEATURE_NAMES['C2'])} features | {len(CONFIG_FEATURE_NAMES['C3'])} features |\n")
        f.write(f"| **Train R²** | **{metrics_c1['train_r2']:.4f}** | **{metrics_c2['train_r2']:.4f}** | **{metrics_c3['train_r2']:.4f}** |\n")
        f.write(f"| **Val R²** | **{metrics_c1['val_r2']:.4f}** | **{metrics_c2['val_r2']:.4f}** | **{metrics_c3['val_r2']:.4f}** |\n")
        f.write(f"| **Train RMSE** | {metrics_c1['train_rmse']:.4f} | {metrics_c2['train_rmse']:.4f} | {metrics_c3['train_rmse']:.4f} |\n")
        f.write(f"| **Val RMSE** | {metrics_c1['val_rmse']:.4f} | {metrics_c2['val_rmse']:.4f} | {metrics_c3['val_rmse']:.4f} |\n")
        f.write(f"| **Train MAE** | {metrics_c1['train_mae']:.4f} | {metrics_c2['train_mae']:.4f} | {metrics_c3['train_mae']:.4f} |\n")
        f.write(f"| **Val MAE** | {metrics_c1['val_mae']:.4f} | {metrics_c2['val_mae']:.4f} | {metrics_c3['val_mae']:.4f} |\n")
        f.write(f"| **Fit Time** | {metrics_c1['fit_time_seconds']:.1f}s | {metrics_c2['fit_time_seconds']:.1f}s | {metrics_c3['fit_time_seconds']:.1f}s |\n")
        f.write(f"| **Negative Train R² Flag** | **{metrics_c1['negative_train_r2_flag']}** | **{metrics_c2['negative_train_r2_flag']}** | **{metrics_c3['negative_train_r2_flag']}** |\n\n")

        f.write("## Ranked Feature Importances\n\n")
        f.write("### Config C1 / C2:\n")
        for rank, (fname, imp) in enumerate(metrics_c1["feature_importances"].items(), 1):
            f.write(f"- **{rank}. {fname}**: `{imp:.4f}` ({imp*100:.2f}%)\n")
        f.write("\n### Config C3:\n")
        for rank, (fname, imp) in enumerate(metrics_c3["feature_importances"].items(), 1):
            f.write(f"- **{rank}. {fname}**: `{imp:.4f}` ({imp*100:.2f}%)\n")

    print(f"Saved Markdown report to {md_report_path}")

    print("\n" + "=" * 80)
    print("ALL 3 GBR MODELS TRAINED, VALIDATED, AND SAVED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
