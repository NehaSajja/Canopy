"""Build train/val/test splits for the expanded geographic baseline.

The original geo baseline (notebook 06) only used the 4,400 observations that
had downloaded images. The expanded baseline reuses the full observation set
in ``artifacts/time_geo_species_df.csv``. To keep results comparable with the
original baseline we must:

  1. Preserve the existing split assignment for every observation already
     listed in ``artifacts/split_iid.csv`` (no leakage between baselines).
  2. Stratify only the *new* observations by species using the same 70/15/15
     ratio and the same random seed (42) used in notebook 06.
  3. Restrict to the same 44 species as the original ``label_encoder.pkl`` /
     ``label_mapping.json`` so the classifier head shape is unchanged.
  4. Apply the same row-level filters as notebook 06 cell 10 so the 4,400
     baseline rows survive unchanged.

Outputs (written to ``ARTIFACTS_DIR``):

  * ``split_geo_expanded.csv``               — split index with provenance
  * ``master_observations_geo_expanded.csv`` — feature-ready table
  * ``split_geo_expanded_manifest.json``     — reproducibility metadata

Usage:

  python scripts/build_expanded_splits.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SCRIPT_VERSION = "1.0.0"
RANDOM_STATE = 42
VAL_FRAC = 0.15
TEST_FRAC = 0.15
GEO_FEATURE_COLS = ["lat_sin", "lat_cos", "lon_sin", "lon_cos", "doy_sin", "doy_cos"]
GEO_TARGET_COL = "label_species"
SPLIT_COL = "split_geo_expanded"

REQUIRED_OBS_COLS = ("gbifID", "species", "decimalLatitude", "decimalLongitude", "doy")


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "artifacts").exists() and (parent / "notebooks").exists():
            return parent
    raise FileNotFoundError("Could not locate repo root from script location.")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _filter_observations(obs_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same filters as notebook 06 cell 10."""
    df = obs_df[list(REQUIRED_OBS_COLS)].copy()
    df = df.dropna(subset=list(REQUIRED_OBS_COLS))
    df = df[df["doy"].between(1, 365)].copy()
    df["gbifID"] = df["gbifID"].astype("int64")
    df["doy"] = df["doy"].astype(int)
    df = df.drop_duplicates(subset=["gbifID"]).reset_index(drop=True)
    return df


def _stratified_three_way_split(
    df: pd.DataFrame,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> pd.Series:
    """Mirror notebook 06 cell 15: 70/15/15 stratified by species.

    Returns a pd.Series indexed like ``df`` with values in {"train","val","test"}.
    Species with fewer than 3 rows can't be stratified; those rows fall back
    to ``train`` (logged as a warning).
    """
    if df.empty:
        return pd.Series(dtype=object, index=df.index, name=SPLIT_COL)

    counts = df["species"].value_counts()
    too_small = counts[counts < 3].index
    fallback_mask = df["species"].isin(too_small)
    fallback_df = df[fallback_mask]
    splittable = df[~fallback_mask]

    if len(too_small):
        print(
            f"  warn: {fallback_mask.sum()} rows across {len(too_small)} species "
            f"have <3 new obs; assigning all to train"
        )

    test_size_first = val_frac + test_frac  # 0.30
    train_part, temp_part = train_test_split(
        splittable,
        test_size=test_size_first,
        stratify=splittable["species"],
        random_state=seed,
    )
    val_part, test_part = train_test_split(
        temp_part,
        test_size=test_frac / test_size_first,  # 0.50
        stratify=temp_part["species"],
        random_state=seed,
    )

    assignments = pd.Series(index=df.index, dtype=object, name=SPLIT_COL)
    assignments.loc[train_part.index] = "train"
    assignments.loc[val_part.index] = "val"
    assignments.loc[test_part.index] = "test"
    assignments.loc[fallback_df.index] = "train"
    return assignments


def _sinusoidal_encode(df: pd.DataFrame) -> pd.DataFrame:
    """Match notebook 06 cell 17 sinusoidal feature encoding."""
    enc = pd.DataFrame(index=df.index)
    lat_norm = df["decimalLatitude"] / 90.0
    lon_norm = df["decimalLongitude"] / 180.0
    doy_norm = (df["doy"] - 183) / 183.0

    enc["lat_sin"] = np.sin(np.pi * lat_norm)
    enc["lat_cos"] = np.cos(np.pi * lat_norm)
    enc["lon_sin"] = np.sin(np.pi * lon_norm)
    enc["lon_cos"] = np.cos(np.pi * lon_norm)
    enc["doy_sin"] = np.sin(np.pi * doy_norm)
    enc["doy_cos"] = np.cos(np.pi * doy_norm)
    return enc


def build_expanded_splits(
    time_geo_path: Path,
    split_iid_path: Path,
    label_mapping_path: Path,
    label_encoder_path: Path,
    out_dir: Path,
    seed: int = RANDOM_STATE,
    val_frac: float = VAL_FRAC,
    test_frac: float = TEST_FRAC,
) -> dict:
    """Build the expanded geo splits and write artifacts to ``out_dir``.

    Returns a summary dict (also persisted to the manifest JSON).
    """
    time_geo_path = Path(time_geo_path)
    split_iid_path = Path(split_iid_path)
    label_mapping_path = Path(label_mapping_path)
    label_encoder_path = Path(label_encoder_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"reading {time_geo_path.name}")
    obs_df_raw = pd.read_csv(time_geo_path, low_memory=False)
    print(f"  raw rows: {len(obs_df_raw):,}")

    print("filtering observations (matching notebook 06 cell 10)")
    obs_df = _filter_observations(obs_df_raw)
    print(f"  after filter: {len(obs_df):,}")

    with open(label_mapping_path) as f:
        label_mapping = json.load(f)
    species_universe = set(label_mapping.values())
    print(f"restricting to {len(species_universe)} baseline species")
    obs_df = obs_df[obs_df["species"].isin(species_universe)].reset_index(drop=True)
    print(f"  after species restriction: {len(obs_df):,}")

    print(f"reading {split_iid_path.name}")
    iid_df = pd.read_csv(split_iid_path)
    iid_df["gbifID"] = iid_df["gbifID"].astype("int64")
    iid_map = dict(zip(iid_df["gbifID"], iid_df["split_iid"]))
    print(f"  baseline split assignments: {len(iid_map):,}")

    obs_df[SPLIT_COL] = obs_df["gbifID"].map(iid_map)
    obs_df["source"] = np.where(
        obs_df[SPLIT_COL].notna(), "iid_baseline", "expanded_new"
    )
    baseline_aligned = int((obs_df["source"] == "iid_baseline").sum())
    new_rows = obs_df[obs_df["source"] == "expanded_new"]
    print(f"  baseline-aligned rows present in time_geo: {baseline_aligned:,}")
    print(f"  new rows needing assignment:               {len(new_rows):,}")

    missing_from_time_geo = len(iid_map) - baseline_aligned
    if missing_from_time_geo:
        # Should be 0 in practice — split_iid is a strict subset of master_observations
        # which itself was derived from time_geo. Surface loudly if it ever isn't.
        raise RuntimeError(
            f"{missing_from_time_geo} rows from split_iid.csv are missing from "
            f"the filtered time_geo set; investigate before continuing."
        )

    print(f"stratified split for {len(new_rows):,} new rows (seed={seed})")
    new_assignments = _stratified_three_way_split(new_rows, val_frac, test_frac, seed)
    obs_df.loc[new_assignments.index, SPLIT_COL] = new_assignments

    assert obs_df[SPLIT_COL].notna().all(), "every row must have a split assignment"
    assert set(obs_df[SPLIT_COL].unique()) <= {"train", "val", "test"}

    # Backward-compatibility check: every gbifID in split_iid keeps its assignment.
    aligned = obs_df[obs_df["source"] == "iid_baseline"]
    mismatches = aligned[
        aligned[SPLIT_COL] != aligned["gbifID"].map(iid_map)
    ]
    if len(mismatches):
        raise RuntimeError(
            f"{len(mismatches)} rows disagree with split_iid.csv assignments"
        )

    print("encoding sinusoidal geo features")
    obs_df[GEO_FEATURE_COLS] = _sinusoidal_encode(obs_df)

    print(f"loading existing label encoder from {label_encoder_path.name}")
    label_encoder = joblib.load(label_encoder_path)
    obs_df[GEO_TARGET_COL] = label_encoder.transform(obs_df["species"]).astype(int)

    split_index = obs_df[["gbifID", "species", SPLIT_COL, "source"]].copy()
    split_index = split_index.sort_values(["gbifID"]).reset_index(drop=True)

    feature_cols = (
        ["gbifID", "species", GEO_TARGET_COL,
         "decimalLatitude", "decimalLongitude", "doy"]
        + GEO_FEATURE_COLS
        + [SPLIT_COL, "source"]
    )
    master_expanded = obs_df[feature_cols].sort_values(["gbifID"]).reset_index(drop=True)

    split_csv = out_dir / "split_geo_expanded.csv"
    master_csv = out_dir / "master_observations_geo_expanded.csv"
    manifest_json = out_dir / "split_geo_expanded_manifest.json"

    split_index.to_csv(split_csv, index=False)
    master_expanded.to_csv(master_csv, index=False)
    print(f"wrote {split_csv.name} ({len(split_index):,} rows)")
    print(f"wrote {master_csv.name} ({len(master_expanded):,} rows)")

    per_split = (
        split_index.groupby(SPLIT_COL)
        .agg(observations=("gbifID", "size"), species_count=("species", "nunique"))
        .to_dict(orient="index")
    )
    per_source_split = (
        split_index.groupby(["source", SPLIT_COL]).size().unstack(fill_value=0).to_dict()
    )

    coverage = (
        split_index.groupby([SPLIT_COL, "species"]).size().unstack(fill_value=0)
    )
    species_missing_in_any_split = [
        s for s in coverage.columns if (coverage[s] == 0).any()
    ]

    summary = {
        "script_version": SCRIPT_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "seed": seed,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "inputs": {
            "time_geo_species_df": {
                "path": str(time_geo_path),
                "sha256": _sha256(time_geo_path),
                "rows": int(len(obs_df_raw)),
            },
            "split_iid": {
                "path": str(split_iid_path),
                "sha256": _sha256(split_iid_path),
                "rows": int(len(iid_df)),
            },
            "label_mapping": {
                "path": str(label_mapping_path),
                "sha256": _sha256(label_mapping_path),
                "species_count": len(species_universe),
            },
            "label_encoder": {
                "path": str(label_encoder_path),
                "sha256": _sha256(label_encoder_path),
            },
        },
        "outputs": {
            "split_geo_expanded": str(split_csv),
            "master_observations_geo_expanded": str(master_csv),
        },
        "row_counts": {
            "total": int(len(split_index)),
            "iid_baseline": baseline_aligned,
            "expanded_new": int(len(new_rows)),
            "per_split": {k: int(v["observations"]) for k, v in per_split.items()},
            "per_source_split": {
                str(k): {str(kk): int(vv) for kk, vv in v.items()}
                for k, v in per_source_split.items()
            },
        },
        "species_missing_in_any_split": species_missing_in_any_split,
    }

    with open(manifest_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {manifest_json.name}")

    print("\nsummary")
    print(f"  total rows:          {summary['row_counts']['total']:,}")
    print(f"  baseline-aligned:    {summary['row_counts']['iid_baseline']:,}")
    print(f"  newly assigned:      {summary['row_counts']['expanded_new']:,}")
    for split_name in ("train", "val", "test"):
        n = summary["row_counts"]["per_split"].get(split_name, 0)
        print(f"  {split_name:>5}: {n:,}")
    if species_missing_in_any_split:
        print(
            f"  WARNING: {len(species_missing_in_any_split)} species missing from "
            "at least one split"
        )

    return summary


def load_split_geo_expanded(
    artifacts_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the expanded splits as (train_df, val_df, test_df).

    Each frame has all feature columns from ``master_observations_geo_expanded.csv``.
    Use ``GEO_FEATURE_COLS`` for X and ``GEO_TARGET_COL`` for y.
    """
    artifacts_dir = Path(artifacts_dir)
    master_csv = artifacts_dir / "master_observations_geo_expanded.csv"
    df = pd.read_csv(master_csv)
    train_df = df[df[SPLIT_COL] == "train"].reset_index(drop=True)
    val_df = df[df[SPLIT_COL] == "val"].reset_index(drop=True)
    test_df = df[df[SPLIT_COL] == "test"].reset_index(drop=True)
    return train_df, val_df, test_df


def main(argv: list[str] | None = None) -> int:
    repo_root = _repo_root()
    artifacts = repo_root / "artifacts"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-geo", type=Path, default=artifacts / "time_geo_species_df.csv")
    parser.add_argument("--split-iid", type=Path, default=artifacts / "split_iid.csv")
    parser.add_argument("--label-mapping", type=Path, default=artifacts / "label_mapping.json")
    parser.add_argument("--label-encoder", type=Path, default=artifacts / "label_encoder.pkl")
    parser.add_argument("--out-dir", type=Path, default=artifacts)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--val-frac", type=float, default=VAL_FRAC)
    parser.add_argument("--test-frac", type=float, default=TEST_FRAC)
    args = parser.parse_args(argv)

    build_expanded_splits(
        time_geo_path=args.time_geo,
        split_iid_path=args.split_iid,
        label_mapping_path=args.label_mapping,
        label_encoder_path=args.label_encoder,
        out_dir=args.out_dir,
        seed=args.seed,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
