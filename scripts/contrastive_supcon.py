"""
contrastive_supcon.py — Supervised Contrastive (SupCon, Khosla 2020) on pair features (4608-D).

Three tasks in one script:
  1) Train a (linear) projection head 4608 → 256 with the SupCon loss
     on ALL 275 pairs (classes: compatible vs incompatible). Produces
     representations for visualisation.
  2) Visualisation: UMAP 256-D → 2-D, two plots
     (color = interaction label, color = NLR phenotype).
  3) Classifier: the same projection head trained under LONO
     (25-fold) with a linear classifier head on top of the projections.
     Output: oof_predictions in the format used by the other model scripts.

Requires the torch and umap-learn packages.

Outputs:
  data/models/eval/supcon_umap_label.png
  data/models/eval/supcon_umap_phenotype.png
  data/models/eval/supcon_projections.npz       — 256-D projections of all 275 pairs
  data/models/oof_predictions_supcon.csv         — LONO predictions
  data/models/fold_summary_supcon.csv
  data/models/eval/supcon_per_phenotype.csv

Usage:
  cd thesis/
  python scripts/contrastive_supcon.py
  python scripts/contrastive_supcon.py --epochs 200 --temperature 0.1
  python scripts/contrastive_supcon.py --skip-classifier   # visualisation only
"""

from __future__ import annotations

import argparse
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


# ─── SupCon loss ────────────────────────────────────────────────────────────

def supcon_loss(features, labels, temperature=0.07):
    """SupCon loss (Khosla 2020).
    features: (B, D) — post-projection embeddings, **L2-normalised**.
    labels: (B,) — class labels (int).
    Returns a scalar loss (mean per anchor)."""
    import torch
    device = features.device
    B = features.size(0)
    # Positive mask: label_i == label_j (excluding the diagonal)
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    # Exclude the diagonal (a sample is not its own positive)
    logits_mask = torch.ones_like(mask) - torch.eye(B, device=device)
    mask = mask * logits_mask
    # Compute logits (similarity matrix)
    sim = torch.matmul(features, features.T) / temperature
    # Numerical stability: subtract the row-wise max
    sim_max, _ = torch.max(sim, dim=1, keepdim=True)
    logits = sim - sim_max.detach()
    # log-prob: log(exp(logits) / sum(exp(logits)))
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
    # Average over positive pairs per anchor (where any exist)
    mask_sum = mask.sum(1)
    valid = mask_sum > 0
    if not valid.any():
        return torch.tensor(0.0, device=device, requires_grad=True)
    mean_log_prob_pos = (mask * log_prob).sum(1)[valid] / mask_sum[valid]
    loss = -mean_log_prob_pos.mean()
    return loss


# ─── Models ─────────────────────────────────────────────────────────────────

def build_projection_head(d_in: int, d_out: int):
    import torch.nn as nn
    return nn.Linear(d_in, d_out, bias=True)


def build_classifier_head(d_in: int):
    import torch.nn as nn
    return nn.Linear(d_in, 2, bias=True)  # 2 classes: incompatible / compatible


# ─── Training routines ───────────────────────────────────────────────────

def standardize_torch(X_train, X_test=None):
    """Standardise (X - mu) / sd using training-set statistics, applied to test as well."""
    import torch
    mu = X_train.mean(dim=0, keepdim=True)
    sd = X_train.std(dim=0, keepdim=True) + 1e-8
    X_train_s = (X_train - mu) / sd
    if X_test is not None:
        X_test_s = (X_test - mu) / sd
        return X_train_s, X_test_s
    return X_train_s


def train_supcon(X, y, args, device):
    """Train the SupCon projection head on the full dataset (for visualisation)."""
    import torch
    import torch.nn.functional as F
    torch.manual_seed(args.seed)

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    X_t = standardize_torch(X_t)

    proj = build_projection_head(X.shape[1], args.proj_dim).to(device)
    opt = torch.optim.Adam(proj.parameters(), lr=args.lr, weight_decay=1e-5)

    n = len(X)
    print(f"  training SupCon (n={n}, d={X.shape[1]}, proj_dim={args.proj_dim}, "
          f"epochs={args.epochs}, batch={args.batch_size}, τ={args.temperature})")
    for epoch in range(args.epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, args.batch_size):
            idx = perm[start:start + args.batch_size]
            xb = X_t[idx]
            yb = y_t[idx]
            zb = proj(xb)
            zb = F.normalize(zb, dim=1)  # L2-normalisation is required by SupCon
            loss = supcon_loss(zb, yb, temperature=args.temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % max(1, args.epochs // 10) == 0:
            print(f"    epoch {epoch+1:>3d}/{args.epochs}  loss={epoch_loss/max(n_batches,1):.4f}")
    proj.eval()
    return proj


def project_all(proj, X, device):
    import torch
    import torch.nn.functional as F
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    X_t = standardize_torch(X_t)
    with torch.no_grad():
        z = proj(X_t)
        z = F.normalize(z, dim=1)
    return z.cpu().numpy()


# ─── Classifier (LONO) ─────────────────────────────────────────────────────

def train_supcon_with_classifier_lono(pack, args, pool_name, device):
    """LONO 25-fold: for each fold, train a projection head + classifier on
    the 24 training NLRs, then predict the 11 pairs of the held-out NLR."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    X = pack["X"].astype(np.float32)
    y = pack["y"].astype(int)
    nlr_id = pack["nlr_id"].astype(str)
    phenotype = pack["phenotype"].astype(str)
    confidence = pack["confidence"].astype(str)
    unique_nlrs = sorted(set(nlr_id))
    n_folds = len(unique_nlrs)
    print(f"\n=== SupCon+Classifier LONO: {pool_name} pool, {n_folds} folds ===")

    proba_oof = np.full(len(y), np.nan, dtype=np.float32)
    pred_oof = np.full(len(y), -1, dtype=np.int8)
    fold_oof = np.full(len(y), "", dtype=object)
    fold_rows = []

    for fold_idx, held_out in enumerate(unique_nlrs, 1):
        test_mask = (nlr_id == held_out)
        train_mask = ~test_mask
        X_tr, X_te = X[train_mask], X[test_mask]
        y_tr = y[train_mask]; y_te = y[test_mask]

        torch.manual_seed(args.seed + fold_idx)
        X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
        X_te_t = torch.tensor(X_te, dtype=torch.float32, device=device)
        y_tr_t = torch.tensor(y_tr, dtype=torch.long, device=device)
        y_te_t = torch.tensor(y_te, dtype=torch.long, device=device)
        # Standardise
        X_tr_t, X_te_t = standardize_torch(X_tr_t, X_te_t)

        # Stage 1: train the SupCon projection head
        proj = build_projection_head(X.shape[1], args.proj_dim).to(device)
        opt_proj = torch.optim.Adam(proj.parameters(), lr=args.lr, weight_decay=1e-5)
        for epoch in range(args.epochs):
            n = len(X_tr_t)
            perm = torch.randperm(n, device=device)
            for start in range(0, n, args.batch_size):
                idx = perm[start:start + args.batch_size]
                xb = X_tr_t[idx]; yb = y_tr_t[idx]
                zb = F.normalize(proj(xb), dim=1)
                loss = supcon_loss(zb, yb, temperature=args.temperature)
                opt_proj.zero_grad()
                loss.backward()
                opt_proj.step()
        # Stage 2: classifier on top of frozen projections
        proj.eval()
        with torch.no_grad():
            z_tr = F.normalize(proj(X_tr_t), dim=1)
            z_te = F.normalize(proj(X_te_t), dim=1)
        clf = build_classifier_head(args.proj_dim).to(device)
        opt_clf = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-4)
        ce = nn.CrossEntropyLoss()
        # Class weights (balanced)
        n_pos = int(y_tr.sum()); n_neg = len(y_tr) - n_pos
        class_w = torch.tensor([len(y_tr)/(2*max(n_neg,1)), len(y_tr)/(2*max(n_pos,1))],
                               dtype=torch.float32, device=device)
        ce_w = nn.CrossEntropyLoss(weight=class_w)
        for epoch in range(args.clf_epochs):
            n = len(z_tr)
            perm = torch.randperm(n, device=device)
            for start in range(0, n, args.batch_size):
                idx = perm[start:start + args.batch_size]
                zb = z_tr[idx]; yb = y_tr_t[idx]
                logits = clf(zb)
                loss = ce_w(logits, yb)
                opt_clf.zero_grad()
                loss.backward()
                opt_clf.step()
        clf.eval()
        with torch.no_grad():
            logits_te = clf(z_te)
            probs = F.softmax(logits_te, dim=1)[:, 1].cpu().numpy()
        preds = (probs >= args.threshold).astype(int)

        proba_oof[test_mask] = probs
        pred_oof[test_mask] = preds
        fold_oof[test_mask] = held_out

        # Per-fold MCC
        from sklearn.metrics import matthews_corrcoef
        if len(np.unique(y_te)) >= 2:
            m = matthews_corrcoef(y_te, preds)
        else:
            m = float("nan")
        fold_rows.append({
            "fold_test_nlr": held_out,
            "phenotype": phenotype[test_mask][0],
            "confidence": confidence[test_mask][0],
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "mcc": m,
            "n_pos": int(y_te.sum()),
            "n_neg": int((y_te == 0).sum()),
        })
        status = "MCC=n/a" if np.isnan(m) else f"MCC={m:.2f}"
        print(f"  [{fold_idx:>2d}/{n_folds}] hold-out {held_out:<30s} "
              f"({phenotype[test_mask][0]}) {status}")
    return proba_oof, pred_oof, fold_oof, pd.DataFrame(fold_rows)


# ─── Wizualizacja UMAP ───────────────────────────────────────────────────────

def make_umap_plots(Z, labels, phenotypes, out_dir: Path, suffix: str):
    """UMAP 256-D → 2-D, two plots."""
    import umap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print(f"  UMAP fitting on Z shape {Z.shape}...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                       metric="cosine", random_state=42)
    Z2 = reducer.fit_transform(Z)
    np.savez(out_dir / f"supcon_umap_coords_{suffix}.npz", Z2=Z2, labels=labels,
             phenotypes=phenotypes)

    # Plot 1: color = interaction label
    fig, ax = plt.subplots(figsize=(8, 6.5))
    neg_mask = labels == 0; pos_mask = labels == 1
    ax.scatter(Z2[neg_mask, 0], Z2[neg_mask, 1], c="#9aa0a6", s=28, alpha=0.55,
               label=f"incompatible (n={neg_mask.sum()})", edgecolor="white", linewidth=0.4)
    ax.scatter(Z2[pos_mask, 0], Z2[pos_mask, 1], c="#d62728", s=44, alpha=0.85,
               label=f"compatible (n={pos_mask.sum()})", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("UMAP-1", fontsize=11)
    ax.set_ylabel("UMAP-2", fontsize=11)
    ax.set_title(f"SupCon → UMAP, {suffix}-pool\n"
                 f"projekcja 256-D → 2-D, kolor: label",
                 fontsize=12, pad=12)
    ax.legend(loc="best", framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / f"supcon_umap_label_{suffix}.png", dpi=150)
    plt.close(fig)

    # Plot 2: color = NLR phenotype
    pheno_colors = {
        "D1": "#1f77b4", "D2": "#ff7f0e", "d3": "#aec7e8",
        "E1": "#d62728", "E2": "#9467bd", "E3": "#2ca02c", "e4": "#c5b0d5",
    }
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for ph in ["d3", "e4", "D2", "D1", "E2", "E1", "E3"]:
        mask = phenotypes == ph
        if not mask.any():
            continue
        ax.scatter(Z2[mask, 0], Z2[mask, 1], c=pheno_colors.get(ph, "gray"),
                   s=34, alpha=0.78, label=f"{ph} (n={mask.sum()})",
                   edgecolor="white", linewidth=0.4)
    ax.set_xlabel("UMAP-1", fontsize=11)
    ax.set_ylabel("UMAP-2", fontsize=11)
    ax.set_title(f"SupCon → UMAP, {suffix}-pool\n"
                 f"projekcja 256-D → 2-D, kolor: phenotype",
                 fontsize=12, pad=12)
    ax.legend(loc="best", framealpha=0.9, fontsize=9, ncol=2)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / f"supcon_umap_phenotype_{suffix}.png", dpi=150)
    plt.close(fig)


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-mean", default=str(ROOT / "data" / "models" / "features_mean.npz"))
    ap.add_argument("--features-max",  default=str(ROOT / "data" / "models" / "features_max.npz"))
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "models"))
    ap.add_argument("--proj-dim", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--clf-epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pool", choices=["mean", "max"], default="mean",
                    help="Which pool to use for visualization + classifier.")
    ap.add_argument("--skip-classifier", action="store_true",
                    help="Visualization only (no LONO classifier).")
    ap.add_argument("--skip-viz", action="store_true",
                    help="Classifier only.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    eval_dir = out_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    import torch
    device = torch.device("mps" if torch.backends.mps.is_available()
                         else "cuda" if torch.cuda.is_available()
                         else "cpu")
    print(f"Device: {device}")

    feat_path = Path(args.features_mean if args.pool == "mean" else args.features_max)
    pack = load_features(feat_path)
    X = pack["X"].astype(np.float32)
    y = pack["y"].astype(int)
    phenos = pack["phenotype"].astype(str)

    # ── (1) Train SupCon on the full dataset + UMAP ─────────────────────────
    if not args.skip_viz:
        print(f"\n=== Stage 1: SupCon training on ALL pairs ({args.pool}-pool) ===")
        proj = train_supcon(X, y, args, device)
        Z = project_all(proj, X, device)
        np.savez(eval_dir / f"supcon_projections_{args.pool}.npz",
                 Z=Z, labels=y, phenotypes=phenos)
        print(f"  → {display_path(eval_dir / f'supcon_projections_{args.pool}.npz')}")
        # UMAP
        try:
            make_umap_plots(Z, y, phenos, eval_dir, args.pool)
            print(f"  → {display_path(eval_dir / f'supcon_umap_label_{args.pool}.png')}")
            print(f"  → {display_path(eval_dir / f'supcon_umap_phenotype_{args.pool}.png')}")
        except ImportError:
            print("  (umap-learn missing — skipping UMAP, projekcje saved)")

    # ── (2) Classifier z LONO ─────────────────────────────────────────────
    if not args.skip_classifier:
        proba, pred, fold, fold_df = train_supcon_with_classifier_lono(
            pack, args, args.pool, device)
        # Only one pool, hence a single set of columns
        oof = pd.DataFrame({
            "nlr_id":     pack["nlr_id"].astype(str),
            "hetc_id":    pack["hetc_id"].astype(str),
            "phenotype":  pack["phenotype"].astype(str),
            "confidence": pack["confidence"].astype(str),
            "label":      pack["y"].astype(int),
            f"pred_prob_{args.pool}":  proba,
            f"pred_label_{args.pool}": pred,
            "fold_test_nlr":   fold,
        })
        oof_path = out_dir / "oof_predictions_supcon.csv"
        oof.to_csv(oof_path, index=False)
        print(f"\nOOF predictions → {display_path(oof_path)}")
        fold_df["pool"] = args.pool
        fold_path = out_dir / "fold_summary_supcon.csv"
        fold_df.to_csv(fold_path, index=False)
        print(f"Fold summary → {display_path(fold_path)}")
        # Per-phenotype
        from sklearn.metrics import matthews_corrcoef, f1_score, roc_auc_score
        prob_col = f"pred_prob_{args.pool}"; pred_col = f"pred_label_{args.pool}"
        m = matthews_corrcoef(oof["label"], oof[pred_col])
        a = roc_auc_score(oof["label"], oof[prob_col])
        f = f1_score(oof["label"], oof[pred_col], zero_division=0)
        print(f"\n=== OVERALL ({args.pool}-pool) ===")
        print(f"  MCC={m:.3f}  F1={f:.3f}  AUC={a:.3f}")

        print("\n=== PER FENOTYP ===")
        print(f"  {'phenotype':<8} {'n':>4} {'n_pos':>6} {'mcc':>8} {'f1':>8} {'auc':>8}")
        pheno_rows = []
        for ph, g in oof.groupby("phenotype"):
            gp = g[prob_col]
            gp_lbl = (gp >= args.threshold).astype(int)
            if len(np.unique(g["label"])) >= 2:
                m_p = matthews_corrcoef(g["label"], gp_lbl)
                a_p = roc_auc_score(g["label"], gp)
            else:
                m_p = float("nan"); a_p = float("nan")
            f_p = f1_score(g["label"], gp_lbl, zero_division=0)
            m_str = f"{m_p:.3f}" if not np.isnan(m_p) else "n/a"
            a_str = f"{a_p:.3f}" if not np.isnan(a_p) else "n/a"
            print(f"  {ph:<8} {len(g):>4d} {int(g['label'].sum()):>6d} "
                  f"{m_str:>8s} {f_p:.3f} {a_str:>8s}")
            pheno_rows.append({"phenotype": ph, "n": len(g),
                              "n_pos": int(g["label"].sum()),
                              "mcc": m_p, "f1": f_p, "auc": a_p})
        pd.DataFrame(pheno_rows).to_csv(eval_dir / "supcon_per_phenotype.csv", index=False)
        print(f"\n→ {display_path(eval_dir / 'supcon_per_phenotype.csv')}")

    print("\nDone.")


if __name__ == "__main__":
    main()
