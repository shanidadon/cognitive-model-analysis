"""Compute ordinal correlations between reaction times and typicality scores.

This module aligns per-image reaction times with typicality scores for a given
combination of neural network model (nn_model) and cognitive categorization
model (categorization_model), then computes ordinal correlations (Spearman and
Kendall).

Assumptions:
- Reaction times file at data/image_reaction_times.csv with columns:
  [category, image_index, mean_rt] (global_image_id optional)
- Typicality file at data/cifar10_individual_typicality_scores (1).csv with
  columns: [dataset, split, category, image_index, nn_model,
  categorization_model, typicality_score]

Outputs:
- results/correlations/merged_<nn>__<cat>.csv: per-image merged data
- results/correlations/stats_<nn>__<cat>.csv: correlation summary
- results/correlations/summary.csv: optional summary across all combos
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from scipy.stats import kendalltau, spearmanr


def load_reaction_times(path: str = "data/image_reaction_times.csv") -> pd.DataFrame:
    """Load reaction times file and validate schema."""
    df = pd.read_csv(path)
    required = {"category", "image_index", "mean_rt"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns in reaction times CSV {path}: {sorted(missing)}"
        )
    return df


def load_typicality(
    path: str = "data/cifar10_individual_typicality_scores (1).csv",
    nn_model: Optional[str] = None,
    categorization_model: Optional[str] = None,
) -> pd.DataFrame:
    """Load typicality scores and filter to dataset=cisar10, split=test, and combo."""
    df = pd.read_csv(path)
    # Filter to CIFAR-10 test split
    df = df[(df["dataset"].str.lower() == "cifar10") & (df["split"].str.lower() == "test")]

    if nn_model is not None:
        df = df[df["nn_model"] == nn_model]
    if categorization_model is not None:
        df = df[df["categorization_model"] == categorization_model]

    # Keep only necessary columns
    cols = [
        "category",
        "image_index",
        "nn_model",
        "categorization_model",
        "typicality_score",
    ]
    return df[cols]


def join_by_combo(rt_df: pd.DataFrame, typ_df: pd.DataFrame) -> pd.DataFrame:
    """Inner-join reaction times and typicality by category and image_index."""
    merged = rt_df.merge(
        typ_df,
        on=["category", "image_index"],
        how="inner",
        validate="one_to_one",
    )
    return merged


def ordinal_corr(merged: pd.DataFrame) -> Dict[str, float]:
    """Compute Spearman and Kendall ordinal correlations."""
    x = merged["typicality_score"]
    y = merged["mean_rt"]
    rho, rho_p = spearmanr(x, y)
    tau, tau_p = kendalltau(x, y)
    return {
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "kendall_tau": float(tau),
        "kendall_p": float(tau_p),
        "n": int(len(merged)),
    }


def compute_for_combo(
    nn_model: str,
    categorization_model: str,
    rt_path: str = "data/image_reaction_times.csv",
    typ_path: str = "data/cifar10_individual_typicality_scores (1).csv",
    out_dir: str = "results/correlations",
) -> Dict[str, float]:
    """Compute merged data and ordinal correlation for one nn/categorization combo."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rt_df = load_reaction_times(rt_path)
    typ_df = load_typicality(typ_path, nn_model, categorization_model)
    merged = join_by_combo(rt_df, typ_df)
    stats = ordinal_corr(merged)

    tag = f"{nn_model}__{categorization_model}".replace(" ", "_")
    merged_out = Path(out_dir) / f"merged_{tag}.csv"
    merged[[
        "category",
        "image_index",
        "mean_rt",
        "typicality_score",
        "nn_model",
        "categorization_model",
    ]].to_csv(merged_out, index=False)

    stats_out = Path(out_dir) / f"stats_{tag}.csv"
    pd.DataFrame([
        {
            "nn_model": nn_model,
            "categorization_model": categorization_model,
            **stats,
        }
    ]).to_csv(stats_out, index=False)

    return stats


def compute_all_combos(
    rt_path: str = "data/image_reaction_times.csv",
    typ_path: str = "data/cifar10_individual_typicality_scores (1).csv",
    out_dir: str = "results/correlations",
) -> pd.DataFrame:
    """Compute correlations for all nn_model × categorization_model combinations."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rt_df = load_reaction_times(rt_path)
    typ_df = load_typicality(typ_path)  # All combos

    results = []
    for (nn, cat), sub in typ_df.groupby(["nn_model", "categorization_model"], sort=True):
        merged = join_by_combo(rt_df, sub)
        stats = ordinal_corr(merged)
        results.append({"nn_model": nn, "categorization_model": cat, **stats})
ן
    summary = pd.DataFrame(results).sort_values(["nn_model", "categorization_model"])  # type: ignore[arg-type]
    (Path(out_dir) / "summary.csv").write_text(summary.to_csv(index=False))
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute ordinal correlation between RT and typicality.")
    parser.add_argument("--nn_model", type=str, help="Neural network model name (e.g., DINOv2_ViT_Large)")
    parser.add_argument(
        "--categorization_model",
        type=str,
        help="Cognitive categorization model (e.g., CIFAR10LDA / prototype / exemplar / decision_bound)",
    ) 
    parser.add_argument("--rt_path", type=str, default="data/image_reaction_times.csv")
    parser.add_argument(
        "--typ_path",
        type=str,
        default="data/cifar10_individual_typicality_scores (1).csv",
    )
    parser.add_argument("--out_dir", type=str, default="results/correlations")
    parser.add_argument("--all", action="store_true", help="Compute for all nn × categorization combos")

    args = parser.parse_args()

    if args.all:
        summary = compute_all_combos(args.rt_path, args.typ_path, args.out_dir)
        print(summary.head())
    else:
        if not args.nn_model or not args.categorization_model:
            raise SystemExit("--nn_model and --categorization_model are required unless --all is specified")
        stats = compute_for_combo(args.nn_model, args.categorization_model, args.rt_path, args.typ_path, args.out_dir)
        print(stats)
