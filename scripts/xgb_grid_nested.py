"""
xgb_grid_nested.py — XGBoost hyperparameter grid search with nested CV.

Outer loop: 25-fold leave-one-NLR-out (model evaluation).
Inner loop: 5-fold grouped by NLR (hyperparameter selection, MCC-optimised).

Grid (108 combinations):
  max_depth ∈ {2, 3, 4, 5}
  min_child_weight ∈ {1, 3, 5}
  learning_rate ∈ {0.03, 0.05, 0.1}
  n_estimators ∈ {200, 400, 600}
Total fits: 25 × 5 × 108 = 13 500.

Requires the xgboost package.

Output:
  data/models/oof_predictions_xgb_tuned.csv
  data/models/fold_summary_xgb_tuned.csv
  data/models/best_params_per_fold.csv
  data/models/grid_full_results.csv

Usage:
  python scripts/xgb_grid_nested.py
  python scripts/xgb_grid_nested.py --inner-folds 3
  python scripts/xgb_grid_nested.py --quick
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def display_path(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p.resolve())


def load_features(path: Path) -> dict:
    npz = np.load(path, allow_pickle=True)
    pack = {k: npz[k] for k in npz.files}
    npz.close()
    return pack


def get_grid(quick: bool = False):
    if quick:
        return {
            "max_depth": [3, 4],
            "min_child_weight": [1, 3],
            "learning_rate": [0.05],
            "n_estimators": [300],
        }
    return {
        "max_depth": [2, 3, 4, 5],
        "min_child_weight": [1, 3, 5],
        "learning_rate": [0.03, 0.05, 0.1],
        "n_estimators": [200, 400, 600],
    }


def grid_combinations(grid: dict):
    keys = list(grid.keys())
    for combo in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, combo))


def build_xgb(params: dict, scale_pos_weight: float, seed: int):
    import xgboost as xgb
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        max_depth=params["max_depth"],
        min_child_weight=params["min_child_weight"],
        subsample=0.8,
        colsample_bytree=0.3,
        learning_rate=params["learning_rate"],
        n_estimators=params["n_estimators"],
        reg_lambda=1.0,
        reg_alpha=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        verbosity=0,
        n_jobs=-1,
    )


def make_inner_folds(nlr_train: np.ndarray, n_folds: int, seed: int):
    """Dzieli unikalne NLR-y w treningu na n_folds grup. Returns liste masek
    (each fold to mask po samples training)."""
    unique = sorted(set(nlr_train.tolist()))
    rng = np.random.RandomState(seed)
    rng.shuffle(unique)
    groups = np.array_split(unique, n_folds)
    folds = []
    for g in groups:
        val_mask = np.isin(nlr_train, list(g))
        folds.append(val_mask)
    return folds


def mcc_np(y, p):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=int)
    tp = int(((p == 1) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    if den == 0:
        return float("nan")
    return (tp * tn - fp * fn) / den


def inner_cv_score(X_tr_outer, y_tr_outer, nlr_tr_outer, params,
                   n_inner_folds, seed, threshold=0.5):
    """Mean MCC over n_inner_folds — used to score a single hyperparameter
    combination."""
    inner_folds = make_inner_folds(nlr_tr_outer, n_inner_folds, seed)
    fold_mccs = []
    for val_mask in inner_folds:
        train_mask = ~val_mask
        y_inner_tr = y_tr_outer[train_mask]
        pos = int(y_inner_tr.sum())
        neg = len(y_inner_tr) - pos
        if pos == 0 or neg == 0:
            continue  # zdegenerowany fold
        spw = neg / max(pos, 1)
        clf = build_xgb(params, scale_pos_weight=spw, seed=seed)
        clf.fit(X_tr_outer[train_mask], y_inner_tr)
        probas = clf.predict_proba(X_tr_outer[val_mask])[:, 1]
        preds = (probas >= threshold).astype(int)
        m = mcc_np(y_tr_outer[val_mask], preds)
        if not np.isnan(m):
            fold_mccs.append(m)
    return float(np.mean(fold_mccs)) if fold_mccs else float("nan")


def run_nested_lono(pack: dict, args, pool_name: str):
    X = pack["X"].astype(np.float32)
    y = pack["y"].astype(int)
    nlr_id = pack["nlr_id"].astype(str)
    phenotype = pack["phenotype"].astype(str)
    confidence = pack["confidence"].astype(str)
    unique_nlrs = sorted(set(nlr_id))
    n_outer = len(unique_nlrs)
    grid = get_grid(args.quick)
    combos = list(grid_combinations(grid))
    print(f"\n=== Nested LONO: {pool_name} pool ===")
    print(f"   outer folds: {n_outer}, inner folds: {args.inner_folds}, "
          f"grid combinations: {len(combos)}")
    print(f"   total fits: {n_outer * args.inner_folds * len(combos) + n_outer}")

    proba_oof = np.full(len(y), np.nan, dtype=np.float32)
    pred_oof = np.full(len(y), -1, dtype=np.int8)
    fold_oof = np.full(len(y), "", dtype=object)
    fold_rows = []
    best_params_rows = []

    # Grid result accumulator (for statistics: how many times each combination
    # was selected as the best inner-CV winner)
    combo_wins = {tuple(sorted(c.items())): 0 for c in combos}

    t_start = time.time()
    for fold_idx, held_out in enumerate(unique_nlrs, 1):
        test_mask = (nlr_id == held_out)
        train_mask = ~test_mask
        X_tr = X[train_mask]; y_tr = y[train_mask]; nlr_tr = nlr_id[train_mask]

        # Inner CV: ocen each combinations
        scores = []
        for combo in combos:
            s = inner_cv_score(X_tr, y_tr, nlr_tr, combo,
                               args.inner_folds, args.seed,
                               args.threshold)
            scores.append((s, combo))
        # Best
        scores.sort(key=lambda x: (np.nan_to_num(x[0], nan=-1), ), reverse=True)
        best_score, best_combo = scores[0]
        combo_wins[tuple(sorted(best_combo.items()))] += 1
        # Trenuj na calym treningu z najlepszymi hiperparametrami
        pos = int(y_tr.sum()); neg = len(y_tr) - pos
        spw = neg / max(pos, 1)
        clf = build_xgb(best_combo, scale_pos_weight=spw, seed=args.seed)
        clf.fit(X_tr, y_tr)
        probas = clf.predict_proba(X[test_mask])[:, 1]
        preds = (probas >= args.threshold).astype(int)
        proba_oof[test_mask] = probas
        pred_oof[test_mask] = preds
        fold_oof[test_mask] = held_out

        m = mcc_np(y[test_mask], preds)
        elapsed = time.time() - t_start
        eta = elapsed / fold_idx * (n_outer - fold_idx)
        status = "MCC=n/a" if np.isnan(m) else f"MCC={m:.2f}"
        print(f"  [{fold_idx:>2d}/{n_outer}] {held_out:<30s} ({phenotype[test_mask][0]}) "
              f"best_inner={best_score:.3f} {status} | "
              f"elapsed={elapsed/60:.1f}min eta={eta/60:.1f}min")

        fold_rows.append({
            "fold_test_nlr": held_out,
            "phenotype": phenotype[test_mask][0],
            "confidence": confidence[test_mask][0],
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "mcc": m,
            "best_inner_mcc": best_score,
            "best_max_depth": best_combo["max_depth"],
            "best_min_child_weight": best_combo["min_child_weight"],
            "best_learning_rate": best_combo["learning_rate"],
            "best_n_estimators": best_combo["n_estimators"],
        })
        best_params_rows.append({
            "fold_test_nlr": held_out,
            **best_combo,
            "best_inner_mcc": best_score,
        })

    # Aggregate: which hyperparameter set wins most often
    grid_summary = []
    for combo in combos:
        wins = combo_wins[tuple(sorted(combo.items()))]
        grid_summary.append({**combo, "n_wins": wins})
    grid_df = pd.DataFrame(grid_summary).sort_values("n_wins", ascending=False)

    return proba_oof, pred_oof, fold_oof, pd.DataFrame(fold_rows), pd.DataFrame(best_params_rows), grid_df


def main():
    ap = argparse.ArgumentParser()
    # Default features: [a, b] concat (2304-D). Ablation showed that explicit
    # interaction terms (|a-b|, a*b) do not improve XGBoost — the trees learn
    # interactions implicitly. Pass features_mean.npz / features_max.npz to
    # tune on the older 4608-D pack instead.
    ap.add_argument("--features-mean",
                    default=str(ROOT / "data" / "models" / "features_concat_mean.npz"))
    ap.add_argument("--features-max",
                    default=str(ROOT / "data" / "models" / "features_concat_max.npz"))
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "models"))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--inner-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true",
                    help="Mala siatka (4 combinations) — do testowania.")
    ap.add_argument("--pool", choices=["mean", "max", "both"], default="both",
                    help="Ktory pool tunowac. 'both' = oba.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pack_mean = load_features(Path(args.features_mean))
    pack_max  = load_features(Path(args.features_max))

    results = {}
    if args.pool in ("mean", "both"):
        results["mean"] = run_nested_lono(pack_mean, args, "mean")
    if args.pool in ("max", "both"):
        results["max"] = run_nested_lono(pack_max, args, "max")

    # Zbierz OOF do jednej tabeli
    if args.pool == "both":
        proba_m, pred_m, fold_m, fold_df_m, best_m, grid_m = results["mean"]
        proba_x, pred_x, fold_x, fold_df_x, best_x, grid_x = results["max"]
        oof = pd.DataFrame({
            "nlr_id":     pack_mean["nlr_id"].astype(str),
            "hetc_id":    pack_mean["hetc_id"].astype(str),
            "phenotype":  pack_mean["phenotype"].astype(str),
            "confidence": pack_mean["confidence"].astype(str),
            "label":      pack_mean["y"].astype(int),
            "pred_prob_mean":  proba_m,
            "pred_prob_max":   proba_x,
            "pred_label_mean": pred_m,
            "pred_label_max":  pred_x,
            "fold_test_nlr":   fold_m,
        })
        fold_df_m["pool"] = "mean"; fold_df_x["pool"] = "max"
        fold_summary = pd.concat([fold_df_m, fold_df_x], ignore_index=True)
        best_m["pool"] = "mean"; best_x["pool"] = "max"
        best_summary = pd.concat([best_m, best_x], ignore_index=True)
        grid_m["pool"] = "mean"; grid_x["pool"] = "max"
        grid_summary = pd.concat([grid_m, grid_x], ignore_index=True)
    else:
        # Single-pool mode
        pool_key = args.pool
        proba, pred, fold, fold_df, best, grid = results[pool_key]
        oof = pd.DataFrame({
            "nlr_id":     pack_mean["nlr_id"].astype(str),
            "hetc_id":    pack_mean["hetc_id"].astype(str),
            "phenotype":  pack_mean["phenotype"].astype(str),
            "confidence": pack_mean["confidence"].astype(str),
            "label":      pack_mean["y"].astype(int),
            f"pred_prob_{pool_key}":  proba,
            f"pred_label_{pool_key}": pred,
            "fold_test_nlr":   fold,
        })
        fold_df["pool"] = pool_key
        fold_summary = fold_df
        best["pool"] = pool_key; best_summary = best
        grid["pool"] = pool_key; grid_summary = grid

    oof_path = out_dir / "oof_predictions_xgb_tuned.csv"
    oof.to_csv(oof_path, index=False)
    print(f"\nOOF predictions → {display_path(oof_path)}")
    fold_path = out_dir / "fold_summary_xgb_tuned.csv"
    fold_summary.to_csv(fold_path, index=False)
    print(f"Fold summary → {display_path(fold_path)}")
    best_path = out_dir / "best_params_per_fold.csv"
    best_summary.to_csv(best_path, index=False)
    print(f"Best params per fold → {display_path(best_path)}")
    grid_path = out_dir / "grid_full_results.csv"
    grid_summary.to_csv(grid_path, index=False)
    print(f"Grid summary → {display_path(grid_path)}")

    # Overall
    if args.pool == "both":
        from sklearn.metrics import matthews_corrcoef, f1_score, roc_auc_score
        print("\n=== OVERALL (po tuningu) ===")
        for col_prob, col_pred, lbl in [
            ("pred_prob_mean", "pred_label_mean", "mean"),
            ("pred_prob_max",  "pred_label_max",  "max"),
        ]:
            m = matthews_corrcoef(oof["label"], oof[col_pred])
            a = roc_auc_score(oof["label"], oof[col_prob])
            f = f1_score(oof["label"], oof[col_pred], zero_division=0)
            print(f"  {lbl:>4s}: MCC={m:.3f}  F1={f:.3f}  AUC={a:.3f}")
        ens = (oof["pred_prob_mean"] + oof["pred_prob_max"]) / 2.0
        ens_lbl = (ens >= args.threshold).astype(int)
        m_e = matthews_corrcoef(oof["label"], ens_lbl)
        a_e = roc_auc_score(oof["label"], ens)
        f_e = f1_score(oof["label"], ens_lbl, zero_division=0)
        print(f"  ensemble: MCC={m_e:.3f}  F1={f_e:.3f}  AUC={a_e:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
